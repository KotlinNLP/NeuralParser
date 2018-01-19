/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.LayerUnit
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.FeedforwardLayersPool
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory

/**
 * The factory of the decoding support structure for attention TPD joint prediction models.
 *
 * @param actionMemoryRNN the action memory RNN
 * @param transformLayerParams the parameters of the transform layers for the attention network
 * @param stateAttentionNetworkParams the parameters of the attention network used to encode the state
 * @param featuresLayerParams the parameters of the features layer
 * @param transitionNetwork the neural network used to score the transitions
 * @param posDeprelNetworkModel the neural model of the multi-task network used to score POS tags and deprels
 * @param outputErrorsInit the default initialization of the output errors
 */
open class AttentionTPDJointStructureFactory(
  private val actionMemoryRNN: NeuralNetwork,
  private val transformLayerParams: FeedforwardLayerParameters,
  private val stateAttentionNetworkParams: AttentionNetworkParameters,
  private val featuresLayerParams: FeedforwardLayerParameters,
  private val transitionNetwork: NeuralNetwork,
  private val posDeprelNetworkModel: MultiTaskNetworkModel,
  private val outputErrorsInit: OutputErrorsInit
) : SupportStructureFactory<AttentionTPDJointSupportStructure> {

  /**
   * Build a new [AttentionTPDJointSupportStructure].
   *
   * @return a new attention decoding support structure
   */
  override fun globalStructure() = AttentionTPDJointSupportStructure(
    actionMemoryEncoder = RecurrentNeuralProcessor(this.actionMemoryRNN),
    transformLayersPool = FeedforwardLayersPool(
      inputType = LayerType.Input.Dense,
      activationFunction = Tanh(),
      params = this.transformLayerParams),
    stateAttentionNetworksPool = AttentionNetworksPool(
      model = this.stateAttentionNetworkParams,
      inputType = LayerType.Input.Dense),
    featuresLayer = FeedforwardLayerStructure(
      inputArray = AugmentedArray(size = this.featuresLayerParams.inputSize),
      outputArray = LayerUnit(size = this.featuresLayerParams.outputSize),
      params = this.featuresLayerParams,
      activationFunction = ReLU()
    ),
    transitionProcessor = FeedforwardNeuralProcessor(this.transitionNetwork),
    posDeprelNetwork = MultiTaskNetwork(this.posDeprelNetworkModel),
    outputErrorsInit = this.outputErrorsInit)
}
