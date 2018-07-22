/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple

import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParserModel
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNN
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator
import com.kotlinnlp.utils.DictionarySet

/**
 * @property langCode the ISO 639-1 language code within the parser works
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property posEmbeddingSize the size of each pos embedding vector
 * @param biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param biRNNLayers number of stacked BiRNNs
 */
abstract class BiRNNParserModel(
  langCode: String,
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  val corpusDictionary: CorpusDictionary,
  val wordEmbeddingSize: Int,
  val posEmbeddingSize: Int,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction?,
  biRNNLayers: Int
) : TransitionBasedParserModel(langCode = langCode, scoreAccumulatorFactory = scoreAccumulatorFactory) {

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
  private val biRNNInputSize: Int = this.posEmbeddingSize + this.wordEmbeddingSize

  /**
   * The part-of-speech embeddings.
   */
  val posEmbeddings = EmbeddingsMapByDictionary(
    dictionary = DictionarySet(this.corpusDictionary.posTags.getElements().map { it.label }),
    size = this.posEmbeddingSize)

  /**
   * The word embeddings.
   */
  val wordEmbeddings = EmbeddingsMapByDictionary(
    dictionary = this.corpusDictionary.words,
    size = this.wordEmbeddingSize)

  /**
   * A [DeepBiRNN] network.
   */
  val deepBiRNN = DeepBiRNN.byReduceConcat(
    inputSize = this.biRNNInputSize,
    hiddenActivation = biRNNHiddenActivation,
    recurrentConnectionType = biRNNConnectionType,
    inputType = LayerType.Input.Dense,
    numberOfLevels = biRNNLayers)
}
