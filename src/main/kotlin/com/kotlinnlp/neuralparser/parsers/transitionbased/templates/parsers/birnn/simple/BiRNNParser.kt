/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple

import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import com.kotlinnlp.syntaxdecoder.GreedyDecoder
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * A [TransitionBasedParser] that uses Embeddings to encode the tokens of a sentence through a BiRNN.
 *
 * If the beamSize is 1 then a [GreedyDecoder] is used, a [BeamDecoder] otherwise.
 *
 * @property model the parser model
 * @param wordDropoutCoefficient the word embeddings dropout coefficient
 * @param posDropoutCoefficient the POS embeddings dropout coefficient
 * @param beamSize the max size of the beam
 * @param maxParallelThreads the max number of threads that can run in parallel (ignored if beamSize is 1)
 */
abstract class BiRNNParser<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  SupportStructureType : DecodingSupportStructure,
  out ModelType: BiRNNParserModel>
(
  model: ModelType,
  private val wordDropoutCoefficient: Double,
  private val posDropoutCoefficient: Double,
  beamSize: Int,
  maxParallelThreads: Int
) :
  TransitionBasedParser<
    StateType,
    TransitionType,
    TokensEmbeddingsContext,
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
  val biRNNEncoder = DeepBiRNNEncoder<DenseNDArray>(model.deepBiRNN)

  /**
   * A [DeepBiRNNEncoder] to encode the a null token
   */
  val paddingVectorEncoder = DeepBiRNNEncoder<DenseNDArray>(model.deepBiRNN)

  /**
   * @param sentence a sentence
   * @param trainingMode a Boolean indicating whether the parser is being trained
   *
   * @return the input context related to the given [sentence]
   */
  override fun buildContext(sentence: Sentence, trainingMode: Boolean): TokensEmbeddingsContext {

    val tokens: List<Token> = sentence.tokens

    val posEmbeddings: List<Embedding> = tokens.map {
      this.model.posEmbeddings.get(element = it.pos!!,
        dropoutCoefficient = if (trainingMode) this.posDropoutCoefficient else 0.0)
    }

    val wordEmbeddings: List<Embedding> = tokens.map {
      this.model.wordEmbeddings.get(element = it.normalizedWord,
        dropoutCoefficient = if (trainingMode) this.wordDropoutCoefficient else 0.0)
    }

    val tokensEmbeddings: List<DenseNDArray> = (0 until tokens.size).map { i ->
      this.buildTokenEmbedding(posEmbedding = posEmbeddings[i], wordEmbedding = wordEmbeddings[i])
    }

    val paddingVector: DenseNDArray = this.paddingVectorEncoder.encode(arrayOf(
      this.buildTokenEmbedding(
        posEmbedding = this.model.posEmbeddings.nullEmbedding,
        wordEmbedding = this.model.wordEmbeddings.nullEmbedding))).first()

    return TokensEmbeddingsContext(
      tokens = tokens,
      items = tokens.map { DenseItem(it.id) },
      posEmbeddings = posEmbeddings,
      wordEmbeddings = wordEmbeddings,
      tokensEncodings = this.biRNNEncoder.encode(sequence = tokensEmbeddings.toTypedArray()),
      encodingSize = this.model.deepBiRNN.outputSize,
      unknownItemVector = paddingVector)
  }

  /**
   * Build the embedding vector associated to a token, given its pos and word embeddings.
   *
   * @param posEmbedding the pos Embedding of the token
   * @param wordEmbedding the word Embedding of the token
   *
   * @return the embedding vector encoding of the given token
   */
  private fun buildTokenEmbedding(posEmbedding: Embedding, wordEmbedding: Embedding): DenseNDArray = concatVectorsV(
    posEmbedding.array.values,
    wordEmbedding.array.values
  )
}
