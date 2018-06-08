/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.charbased

import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParserModel
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.simplednn.core.arrays.UpdatableDenseArray
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.FixedRangeRandom
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attention.han.HAN
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNN
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property charEmbeddingSize the size of each char embedding vector
 * @property charsEncodingSize the size of each char encoding vector (the encoding of more char embeddings)
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property hanAttentionSize the size of the attention arrays
 * @property hanConnectionType the recurrent connection type of the HAN
 * @property hanHiddenActivation the hidden activation function of the HAN BiRNN networks
 * @property biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @property biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 */
abstract class CharBasedBiRNNParserModel(
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  val corpusDictionary: CorpusDictionary,
  val charEmbeddingSize: Int,
  val charsEncodingSize: Int,
  val wordEmbeddingSize: Int,
  val hanAttentionSize: Int,
  val hanConnectionType: LayerType.Connection,
  val hanHiddenActivation: ActivationFunction,
  val biRNNConnectionType: LayerType.Connection,
  val biRNNHiddenActivation: ActivationFunction
) : TransitionBasedParserModel(scoreAccumulatorFactory) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The input size of the [deepBiRNN].
   */
  private val biRNNInputSize: Int = this.charsEncodingSize + this.wordEmbeddingSize

  /**
   * The word embeddings.
   */
  val wordEmbeddings = EmbeddingsMapByDictionary(
    dictionary = this.corpusDictionary.words,
    size = this.wordEmbeddingSize
  )

  /**
   * The chars embeddings.
   */
  val charsEmbeddings = EmbeddingsMap<Char>(size = this.charEmbeddingSize)

  /**
   * A [HAN] model.
   */
  val han = HAN(
    hierarchySize = 1,
    inputSize = this.charEmbeddingSize,
    inputType = LayerType.Input.Dense,
    biRNNsActivation = this.hanHiddenActivation,
    biRNNsConnectionType = this.hanConnectionType,
    attentionSize = this.hanAttentionSize,
    outputSize = this.charsEncodingSize,
    outputActivation = null,
    gainFactors = listOf(2.0))

  /**
   * A [DeepBiRNN] network.
   */
  val deepBiRNN = DeepBiRNN(
    inputSize = this.biRNNInputSize,
    hiddenActivation = this.biRNNHiddenActivation,
    recurrentConnectionType = this.biRNNConnectionType,
    inputType = LayerType.Input.Dense,
    numberOfLayers = 1)

  /**
   * The embedding used to represent the encoding of a null item of the decoding window.
   */
  val unknownItemEmbedding = Embedding(
    id = -1,
    array = UpdatableDenseArray(Shape(this.deepBiRNN.outputSize)))

  /**
   * Initialize chars embeddings and the nullItemVector.
   */
  init {

    this.unknownItemEmbedding.array.values.randomize(FixedRangeRandom(radius = 0.08, enablePseudoRandom = true))

    this.corpusDictionary.words.getElements().forEach {
      it.forEach { char ->
        if (!this.charsEmbeddings.contains(char)) this.charsEmbeddings.set(char)
      }
    }
  }
}
