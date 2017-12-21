/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint.actionsembeddings.ActionsEmbeddingsMap
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ActionsScorerNetworkBuilder
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos.BiRNNAmbiguousPOSParserModel
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkConfig
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * The parser model for the ArcStandard parser based on the BiRNN with Action+Transition+POS+Deprel joint scoring.
 *
 * @property actionsScoresActivation the function used to activate the actions scores (can be null)
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property unknownFormDefaultPOSTags the list of POS tags to use for unknown forms
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property posEmbeddingSize the size of each POS embedding vector
 * @property actionsEmbeddingsSize the size of each action embedding vector
 * @property preTrainedWordEmbeddings pre-trained word embeddings to add to the tokens encodings (default = null)
 * @param biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param biRNNLayers number of stacked BiRNNs
 * @param appliedActionsNetworkConfig the configuration of the network used to encode the last applied actions
 * @param scorerNetworksConfig the configuration of the scorer networks
 */
class BiRNNATPDJointArcStandardParserModel(
  val actionsScoresActivation: ActivationFunction?,
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  corpusDictionary: CorpusDictionary,
  unknownFormDefaultPOSTags: List<POSTag>,
  wordEmbeddingSize: Int,
  posEmbeddingSize: Int,
  val actionsEmbeddingsSize: Int,
  preTrainedWordEmbeddings: EmbeddingsMap<String>? = null,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction?,
  biRNNLayers: Int,
  appliedActionsNetworkConfig: AppliedActionsNetworkConfiguration,
  scorerNetworksConfig: ScorerNetworkConfiguration
) : BiRNNAmbiguousPOSParserModel(
  scoreAccumulatorFactory = scoreAccumulatorFactory,
  corpusDictionary = corpusDictionary,
  unknownFormDefaultPOSTags = unknownFormDefaultPOSTags,
  wordEmbeddingSize = wordEmbeddingSize,
  posEmbeddingSize = posEmbeddingSize,
  preTrainedWordEmbeddings = preTrainedWordEmbeddings,
  biRNNConnectionType = biRNNConnectionType,
  biRNNHiddenActivation = biRNNHiddenActivation,
  biRNNLayers = biRNNLayers
) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The configuration of the recurrent [NeuralNetwork] used to encode the last applied actions.
   *
   * @property hiddenSize the hidden layer size
   * @property hiddenConnectionType the type of recurrent connection of the hidden layer
   * @property hiddenActivation the hidden layer activation function
   * @property hiddenDropout the probability of dropout of the hidden layer (default = 0.0)
   * @property hiddenMeProp whether to use the 'meProp' errors propagation algorithm for the hidden layer (default false)
   * @property outputSize the output layer size
   * @property outputMeProp whether to use the 'meProp' errors propagation algorithm for the output layer (default false)
   */
  data class AppliedActionsNetworkConfiguration(
    val hiddenSize: Int,
    val hiddenConnectionType: LayerType.Connection,
    val hiddenActivation: ActivationFunction,
    val hiddenDropout: Double = 0.0,
    val hiddenMeProp: Boolean = false,
    val outputSize: Int,
    val outputMeProp: Boolean = false) {

    /**
     * Check the connection type.
     */
    init {
      require(this.hiddenConnectionType.property == LayerType.Property.Recurrent) {
        "The hidden connection type of the network that encodes the last applied actions must be Recurrent."
      }
    }
  }

  /**
   * The number of deprels in the corpus dictionary.
   */
  private val deprelsCount = this.corpusDictionary.deprelTags.size

  /**
   * The number of POS tags in the corpus dictionary.
   */
  private val posTagsCount = this.corpusDictionary.posTags.size

  /**
   * The actions embeddings map.
   */
  val actionsEmbeddings = ActionsEmbeddingsMap(
    size = this.actionsEmbeddingsSize,
    transitionsSize = 4,
    posTagsSize = this.posTagsCount + 1, // + shift offset
    deprelsSize = this.deprelsCount + 1) // + shift offset

  /**
   * The neural network that encodes the last applied actions.
   */
  val appliedActionsNetwork: NeuralNetwork = this.buildAppliedActionsNetwork(appliedActionsNetworkConfig)

  /**
   * The neural network of the actions scorer that scores the transition.
   */
  val transitionScorerNetwork: NeuralNetwork = ActionsScorerNetworkBuilder(
    inputSize = appliedActionsNetworkConfig.outputSize + 4 * this.biRNN.outputSize, // a tokens window of 4 elements
    inputType = LayerType.Input.Dense,
    outputSize = 4, // Shift, Root, ArcLeft, ArcRight
    scorerNetworkConfig = scorerNetworksConfig
  )

  /**
   * The model of the neural network of the actions scorer that scores POS tag and deprel.
   */
  val posDeprelScorerNetworkModel = MultiTaskNetworkModel(
    inputSize = appliedActionsNetworkConfig.outputSize + 4 * this.biRNN.outputSize, // a tokens window of 4 elements
    inputType = LayerType.Input.Dense,
    inputDropout = scorerNetworksConfig.inputDropout,
    hiddenSize = scorerNetworksConfig.hiddenSize,
    hiddenActivation = scorerNetworksConfig.hiddenActivation,
    hiddenDropout = scorerNetworksConfig.hiddenDropout,
    hiddenMeProp = scorerNetworksConfig.hiddenMeProp,
    outputConfigurations = listOf(
      MultiTaskNetworkConfig( // POS tags scoring network
        outputSize = this.posTagsCount + 1, // POS tags + Shift
        outputActivation = scorerNetworksConfig.outputActivation,
        outputMeProp = scorerNetworksConfig.outputMeProp
      ),
      MultiTaskNetworkConfig( // Deprels scoring network
        outputSize = this.deprelsCount + 1, // Deprels + Shift
        outputActivation = scorerNetworksConfig.outputActivation,
        outputMeProp = scorerNetworksConfig.outputMeProp
      )
    )
  )

  /**
   * @param appliedActionsNetworkConfig
   *
   * @return the [NeuralNetwork] to encode the last applied actions
   */
  private fun buildAppliedActionsNetwork(appliedActionsNetworkConfig: AppliedActionsNetworkConfiguration) =
    NeuralNetwork(
      LayerConfiguration(
        size = this.actionsEmbeddingsSize,
        inputType = LayerType.Input.Dense
      ),
      LayerConfiguration(
        size = appliedActionsNetworkConfig.hiddenSize,
        activationFunction = appliedActionsNetworkConfig.hiddenActivation,
        connectionType = appliedActionsNetworkConfig.hiddenConnectionType,
        dropout = appliedActionsNetworkConfig.hiddenDropout,
        meProp = appliedActionsNetworkConfig.hiddenMeProp
      ),
      LayerConfiguration(
        size = appliedActionsNetworkConfig.outputSize,
        activationFunction = null,
        connectionType = LayerType.Connection.Feedforward,
        meProp = appliedActionsNetworkConfig.outputMeProp
      )
    )
}
