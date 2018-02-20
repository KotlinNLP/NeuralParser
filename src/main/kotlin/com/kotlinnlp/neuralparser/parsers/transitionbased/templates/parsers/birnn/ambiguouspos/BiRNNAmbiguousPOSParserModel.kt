/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.ambiguouspos

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParserModel
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNN
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * The model of the [BiRNNAmbiguousPOSParser].
 *
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property nounDefaultPOSTag the 'Noun' POS tag used for title case unknown forms
 * @property otherDefaultPOSTags the list of POS tags used for other unknown forms
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property posEmbeddingSize the size of each pos embedding vector
 * @property preTrainedWordEmbeddings pre-trained word embeddings to add to the tokens encodings (can be null)
 * @param biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param biRNNLayers number of stacked BiRNNs
 */
abstract class BiRNNAmbiguousPOSParserModel(
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  val corpusDictionary: CorpusDictionary,
  val nounDefaultPOSTag: POSTag,
  val otherDefaultPOSTags: List<POSTag>,
  val wordEmbeddingSize: Int,
  val posEmbeddingSize: Int,
  val preTrainedWordEmbeddings: EmbeddingsMap<String>?,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction?,
  biRNNLayers: Int
) : TransitionBasedParserModel(scoreAccumulatorFactory) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The number of all possible POS tags.
   */
  val posTagsSize: Int = this.corpusDictionary.posTags.size

  /**
   * The list of POS tags used for unknown forms.
   */
  val unknownFormDefaultPOSTags: List<POSTag> =
    this.otherDefaultPOSTags.toSet().union(setOf(this.nounDefaultPOSTag)).toList() // a list with unique elements

  /**
   * The input size of the [deepBiRNN].
   */
  private val biRNNInputSize: Int =
    this.posTagsSize + this.wordEmbeddingSize + (this.preTrainedWordEmbeddings?.size ?: 0)

  /**
   * A dictionary set of all possible POS tags labels.
   */
  private val posTagsLabels: DictionarySet<String> = DictionarySet(
    elements = this.corpusDictionary.posTags.getElements().map { it.label }
  )

  /**
   * The word embeddings.
   */
  val wordEmbeddings = EmbeddingsMapByDictionary(
    dictionary = this.corpusDictionary.words,
    size = this.wordEmbeddingSize)

  /**
   * A map of POS tags labels to indices.
   */
  val posTagsToIndices: Map<String, Int> = mapOf(
    *this.posTagsLabels.getElements()
      .map { Pair(it, this.posTagsLabels.getId(it)!!) }
      .toTypedArray()
  )

  /**
   * A [DeepBiRNN] network.
   */
  val deepBiRNN = DeepBiRNN(
    inputSize = this.biRNNInputSize,
    hiddenActivation = biRNNHiddenActivation,
    recurrentConnectionType = biRNNConnectionType,
    inputType = LayerType.Input.Dense,
    numberOfLayers = biRNNLayers)
}
