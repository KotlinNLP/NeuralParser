/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.charbased

import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensCharsEncodingContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HANEncodersPool
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han.HierarchySequence
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
 * A [TransitionBasedParser] that uses Embeddings encoded with a BiRNN to encode the tokens of a sentence.
 *
 * If the beamSize is 1 then a [GreedyDecoder] is used, a [BeamDecoder] otherwise.
 *
 * @property model the parser model
 * @param wordDropoutCoefficient the word embeddings dropout coefficient
 * @param beamSize the max size of the beam
 * @param maxParallelThreads the max number of threads that can run in parallel (ignored if beamSize is 1)
 */
abstract class CharBasedBiRNNParser<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  SupportStructureType : DecodingSupportStructure,
  out ModelType: CharBasedBiRNNParserModel>
(
  model: ModelType,
  private val wordDropoutCoefficient: Double,
  beamSize: Int,
  maxParallelThreads: Int
) :
  TransitionBasedParser<
    StateType,
    TransitionType,
    TokensCharsEncodingContext,
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
   * A [HANEncodersPool] to encode the chars of a token.
   */
  private val hanEncodersPool = HANEncodersPool<DenseNDArray>(model = this.model.han)

  /**
   * @param sentence a sentence
   * @param trainingMode a Boolean indicating whether the parser is being trained
   *
   * @return the input context related to the given [sentence]
   */
  override fun buildContext(sentence: Sentence, trainingMode: Boolean): TokensCharsEncodingContext {

    val tokens: List<Token> = sentence.tokens

    val wordEmbeddings: List<Embedding> = tokens.map {
      this.model.wordEmbeddings.get(
        element = it.normalizedWord, dropoutCoefficient = if (trainingMode) this.wordDropoutCoefficient else 0.0)
    }

    val charsEmbeddings: List<List<Embedding>> = tokens.map {
      it.normalizedWord.map { char -> this.model.charsEmbeddings.get(char) }
    }

    val (usedEncoders, charsEncodings) = this.encodeTokensByChars(tokens = tokens, charsEmbeddings = charsEmbeddings)

    return TokensCharsEncodingContext(
      tokens = tokens,
      items = tokens.map { DenseItem(it.id) },
      usedEncoders = usedEncoders,
      charsEmbeddings = charsEmbeddings,
      wordEmbeddings = wordEmbeddings,
      tokensEncodings = this.biRNNEncoder.encode(sequence = Array(
        size = tokens.size,
        init = { i -> this.buildTokenEmbedding(charsEncoding = charsEncodings[i], wordEmbedding = wordEmbeddings[i]) }
      )),
      encodingSize = this.model.biRNN.outputSize,
      nullItemVector = this.model.unknownItemEmbedding.array.values // TODO
    )
  }

  /**
   * Encode tokens by chars using a HAN with one level.
   *
   * @param tokens the list of tokens of a sentence
   * @param charsEmbeddings the list of lists of chars embeddings, one per token
   *
   * @return a pair with the list of used encoders and the parallel list of tokens encodings by chars
   */
  private fun encodeTokensByChars(
    tokens: List<Token>,
    charsEmbeddings: List<List<Embedding>>
  ): Pair<List<HANEncoder<DenseNDArray>>, List<DenseNDArray>> {

    this.hanEncodersPool.releaseAll()

    val usedEncoders = mutableListOf<HANEncoder<DenseNDArray>>()

    val charsEncodings: List<DenseNDArray> = List(
      size = tokens.size,
      init = { i ->

        val charsVectors = HierarchySequence(*charsEmbeddings[i].map { it.array.values }.toTypedArray())

        usedEncoders.add(this.hanEncodersPool.getItem())

        usedEncoders[i].forward(charsVectors)
      })

    return Pair(usedEncoders, charsEncodings)
  }

  /**
   * Build the embedding vector associated to a token, given its chars and word embeddings.
   *
   * @param charsEncoding the chars encoding vector of a word
   * @param wordEmbedding the word Embedding of the token
   *
   * @return the embedding vector encoding of the given token
   */
  private fun buildTokenEmbedding(charsEncoding: DenseNDArray, wordEmbedding: Embedding): DenseNDArray = concatVectorsV(
    charsEncoding,
    wordEmbedding.array.values
  )
}
