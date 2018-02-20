/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple.BiRNNParserModel
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionModel
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * The parser model for the ArcEagerSpine parser based on the BiRNN.
 *
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property posEmbeddingSize the size of each pos embedding vector
 * @param biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param biRNNLayers number of stacked BiRNNs
 * @param scorerNetworksConfig the configuration of the scorer networks
 */
class BiRNNArcEagerSpineParserModel(
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  corpusDictionary: CorpusDictionary,
  wordEmbeddingSize: Int,
  posEmbeddingSize: Int,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction?,
  biRNNLayers: Int,
  scorerNetworksConfig: ScorerNetworkConfiguration
) : BiRNNParserModel(
  scoreAccumulatorFactory = scoreAccumulatorFactory,
  corpusDictionary = corpusDictionary,
  wordEmbeddingSize = wordEmbeddingSize,
  posEmbeddingSize = posEmbeddingSize,
  biRNNConnectionType = biRNNConnectionType,
  biRNNHiddenActivation = biRNNHiddenActivation,
  biRNNLayers = biRNNLayers) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The neural network of the actions scorer.
   */
  val actionsScorerNetworkModel: MultiPredictionModel

  /**
   * Initialize the actionsScorerNetworkModel.
   */
  init {

    val biRNNOutput: Int = this.deepBiRNN.outputSize
    val sortedDeprels: Set<Deprel> = corpusDictionary.deprelTags.getElementsReversedSet()
    val rootDeprelsCount: Int = sortedDeprels.filter { it.direction == Deprel.Position.ROOT }.count()
    val leftDeprelsCount: Int = sortedDeprels.filter { it.direction == Deprel.Position.LEFT }.count()
    val rightDeprelsCount: Int = sortedDeprels.filter { it.direction == Deprel.Position.RIGHT }.count()

    this.actionsScorerNetworkModel = MultiPredictionModel(
      this.buildMPNetworkConfig( // Shift
        scorerNetworksConfig = scorerNetworksConfig, inputSize = 2 * biRNNOutput, outputSize = 1),
      this.buildMPNetworkConfig( // Root
        scorerNetworksConfig = scorerNetworksConfig, inputSize = 1 * biRNNOutput, outputSize = rootDeprelsCount),
      this.buildMPNetworkConfig( // ArcLeft
        scorerNetworksConfig = scorerNetworksConfig, inputSize = 2 * biRNNOutput, outputSize = leftDeprelsCount),
      this.buildMPNetworkConfig( // ArcRight
        scorerNetworksConfig = scorerNetworksConfig, inputSize = 2 * biRNNOutput, outputSize = rightDeprelsCount)
    )
  }
}
