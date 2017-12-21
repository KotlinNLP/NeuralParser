/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import com.kotlinnlp.syntaxdecoder.GreedyDecoder
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * A [NeuralParser] that uses Embeddings and ambiguous POS vectors to encode the tokens of a sentence through a BiRNN.
 *
 * If the beamSize is 1 then a [GreedyDecoder] is used, a [BeamDecoder] otherwise.
 *
 * @property model the parser model
 * @param wordDropoutCoefficient the word embeddings dropout coefficient
 * @param beamSize the max size of the beam
 * @param maxParallelThreads the max number of threads that can run in parallel (ignored if beamSize is 1)
 */
abstract class BiRNNAmbiguousPOSParser<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  SupportStructureType : DecodingSupportStructure,
  out ModelType: BiRNNAmbiguousPOSParserModel>
(
  model: ModelType,
  private val wordDropoutCoefficient: Double,
  beamSize: Int,
  maxParallelThreads: Int
) :
  NeuralParser<
    StateType,
    TransitionType,
    TokensAmbiguousPOSContext,
    DenseItem,
    FeaturesErrorsType,
    FeaturesType,
    SupportStructureType,
    ModelType>
  (
    model = model,
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads
  ) {

  /**
   * A [DeepBiRNNEncoder] to encode the tokens of a sentence.
   */
  val biRNNEncoder = DeepBiRNNEncoder<DenseNDArray>(this.model.biRNN)

  /**
   * A [DeepBiRNNEncoder] to encode the a null token
   */
  val paddingVectorEncoder = DeepBiRNNEncoder<DenseNDArray>(this.model.biRNN)

  /**
   * @param sentence a sentence
   * @param trainingMode a Boolean indicating whether the parser is being trained
   *
   * @return the input context related to the given [sentence]
   */
  override fun buildContext(sentence: Sentence, trainingMode: Boolean): TokensAmbiguousPOSContext {

    val tokens: List<Token> = sentence.tokens
    val posTags: List<List<POSTag>> = tokens.map {
      when {

        this.model.corpusDictionary.formsToPosTags.containsKey(it.normalizedWord) ->
          this.model.corpusDictionary.formsToPosTags[it.normalizedWord].toList()

        it.normalizedWord.isTitleCase() -> listOf(this.model.nounDefaultPOSTag)

        else -> this.model.unknownFormDefaultPOSTags
      }
    }

    val wordEmbeddings: List<Embedding> = tokens.map {
      this.model.wordEmbeddings.get(
        element = it.normalizedWord,
        dropoutCoefficient = if (trainingMode) this.wordDropoutCoefficient else 0.0)
    }

    val preTrainedEmbeddings: List<Embedding>? = if (this.model.preTrainedWordEmbeddings != null)
      tokens.map { this.model.preTrainedWordEmbeddings.get(it.normalizedWord) }
    else
      null

    val tokensEmbeddings: List<DenseNDArray> = (0 until tokens.size).map { i ->
      this.buildTokenEmbedding(
        posVector = this.buildAmbiguousPOSVector(posTags[i]),
        wordEmbedding = wordEmbeddings[i],
        preTrainedWordEmbedding = preTrainedEmbeddings?.get(i))
    }

    val paddingVector: DenseNDArray = this.paddingVectorEncoder.encode(arrayOf(
      this.buildTokenEmbedding(
        posVector = DenseNDArrayFactory.zeros(Shape(this.model.posTagsSize)),
        wordEmbedding = this.model.wordEmbeddings.nullEmbedding,
        preTrainedWordEmbedding = this.model.preTrainedWordEmbeddings?.nullEmbedding))).first()

    return TokensAmbiguousPOSContext(
      tokens = tokens,
      items = tokens.map { DenseItem(it.id) },
      posTags = posTags,
      wordEmbeddings = wordEmbeddings,
      preTrainedWordEmbeddings = preTrainedEmbeddings,
      tokensEncodings = this.biRNNEncoder.encode(sequence = tokensEmbeddings.toTypedArray()),
      encodingSize = this.model.biRNN.outputSize,
      unknownItemVector = paddingVector,
      trainingMode = trainingMode)
  }

  /**
   * @param posTags the list of ambiguous POS tag vectors of a token
   *
   * @return the vector representation of the given [posTags] (in a dense array with binary values)
   */
  private fun buildAmbiguousPOSVector(posTags: List<POSTag>): DenseNDArray {

    val vector: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.posTagsSize))

    posTags.forEach {
      val posTagIndex: Int = this.model.posTagsToIndices[it.label]!!
      vector[posTagIndex] = 1.0 // activate values corresponding to ambiguous POS tags
    }

    return vector
  }

  /**
   * Build the embedding vector associated to a token, given its pos tags vector, word embedding and an optional
   * pre-trained word embedding.
   *
   * @param posVector the binary vector representing the ambiguous POS tags the token
   * @param wordEmbedding the word embedding of the token,
   * @param preTrainedWordEmbedding the pre-trained word embedding of the token (can be null)
   *
   * @return the embedding vector encoding of the given token
   */
  private fun buildTokenEmbedding(posVector: DenseNDArray,
                                  wordEmbedding: Embedding,
                                  preTrainedWordEmbedding: Embedding?): DenseNDArray =
    if (preTrainedWordEmbedding != null)
      concatVectorsV(posVector, wordEmbedding.array.values, preTrainedWordEmbedding.array.values)
    else
      concatVectorsV(posVector, wordEmbedding.array.values)

  /**
   * Whether a string is title case: first char upper case and all the others lower case.
   *
   * @return a Boolean indicating if this string is title case
   */
  private fun String.isTitleCase(): Boolean =
    this.first().isUpperCase() && this.substring(1, this.length).all { it.isLowerCase() }
}
