/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.attention.tpdjoint

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ActionsScorerNetworkBuilder
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos.BiRNNAmbiguousPOSParserModel
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkConfig
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.simplednn.deeplearning.attentiverecurrentnetwork.AttentiveRecurrentNetworkModel
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * The parser model for the ArcStandard parser based on the BiRNN with Transition+POS+Deprel joint scoring.
 *
 * @property actionsScoresActivation the function used to activate the actions scores (can be null)
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property nounDefaultPOSTag the 'Noun' POS tag used for title case unknown forms
 * @property otherDefaultPOSTags the list of POS tags used for other unknown forms
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property posEmbeddingSize the size of each POS embedding vector
 * @property preTrainedWordEmbeddings pre-trained word embeddings to add to the tokens encodings (default = null)
 * @param biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param biRNNLayers number of stacked BiRNNs
 * @param scorerNetworksConfig the configuration of the scorer networks
 */
class BiRNNAttentionTPDJointArcStandardParserModel(
  val actionsScoresActivation: ActivationFunction?,
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  corpusDictionary: CorpusDictionary,
  nounDefaultPOSTag: POSTag,
  otherDefaultPOSTags: List<POSTag>,
  wordEmbeddingSize: Int,
  posEmbeddingSize: Int,
  preTrainedWordEmbeddings: EmbeddingsMap<String>? = null,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction?,
  biRNNLayers: Int,
  scorerNetworksConfig: ScorerNetworkConfiguration,
  attentionSize: Int,
  tpdSingleEmbeddingSize: Int,
  featuresSize: Int
) : BiRNNAmbiguousPOSParserModel(
  scoreAccumulatorFactory = scoreAccumulatorFactory,
  corpusDictionary = corpusDictionary,
  nounDefaultPOSTag = nounDefaultPOSTag,
  otherDefaultPOSTags = otherDefaultPOSTags,
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
   * The number of deprels in the corpus dictionary.
   */
  private val deprelsCount = this.corpusDictionary.deprelTags.size

  /**
   * The number of POS tags in the corpus dictionary.
   */
  private val posTagsCount = this.corpusDictionary.posTags.size

  /**
   * The actions encoding vectors map.
   */
  val actionsVectors = ActionsVectorsMap(
    size = tpdSingleEmbeddingSize,
    transitionsSize = 4,
    posTagsSize = this.posTagsCount + 1, // + shift offset
    deprelsSize = this.deprelsCount + 1) // + shift offset

  /**
   *
   */
  val attentiveRecurrentNetworkModel = AttentiveRecurrentNetworkModel(
    inputSize = this.biRNN.outputSize,
    attentionSize = attentionSize,
    recurrentContextSize = this.biRNN.outputSize,
    contextLabelSize = tpdSingleEmbeddingSize * 3, // transition, pos, deprel
    outputSize = featuresSize,
    contextActivation = Tanh(),
    contextRecurrenceType = LayerType.Connection.LSTM,
    outputActivationFunction = Tanh())

  /**
   * The neural network of the actions scorer that scores the transition.
   */
  val transitionScorerNetwork: NeuralNetwork = ActionsScorerNetworkBuilder(
    inputSize = this.attentiveRecurrentNetworkModel.outputSize, // the input is a tokens window of 4 elements
    inputType = LayerType.Input.Dense,
    outputSize = 4, // Shift, Root, ArcLeft, ArcRight
    scorerNetworkConfig = scorerNetworksConfig
  )

  /**
   * The model of the neural network of the actions scorer that scores POS tag and deprel.
   */
  val posDeprelScorerNetworkModel = MultiTaskNetworkModel(
    inputSize = this.attentiveRecurrentNetworkModel.outputSize, // the input is a tokens window of 4 elements
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
}
