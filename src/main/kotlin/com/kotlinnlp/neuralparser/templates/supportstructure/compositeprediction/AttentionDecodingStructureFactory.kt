/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory

/**
 * The factory of the decoding support structure for attention decoding prediction models.
 *
 * @param actionDecodingNetwork the recurrent neural network used to decode the action features
 * @param actionAttentionNetworkParams the parameters of the attention network used to decode the action features
 * @param transitionNetwork the neural network used to score the transitions
 * @param posDeprelNetworkModel the neural model of the multi-task network used to score POS tags and deprels
 * @param outputErrorsInit the default initialization of the output errors
 */
open class AttentionDecodingStructureFactory(
  private val actionDecodingNetwork: NeuralNetwork,
  private val actionAttentionNetworkParams: AttentionNetworkParameters,
  private val transitionNetwork: NeuralNetwork,
  private val posDeprelNetworkModel: MultiTaskNetworkModel,
  private val outputErrorsInit: OutputErrorsInit
) : SupportStructureFactory<AttentionDecodingSupportStructure> {

  /**
   * Build a new [AttentionDecodingSupportStructure].
   *
   * @return a new attention decoding support structure
   */
  override fun globalStructure() = AttentionDecodingSupportStructure(
    actionDecoder = RecurrentNeuralProcessor(this.actionDecodingNetwork),
    attentionNetworksPool = AttentionNetworksPool(
      model = this.actionAttentionNetworkParams,
      inputType = LayerType.Input.Dense),
    transitionProcessor = FeedforwardNeuralProcessor(this.transitionNetwork),
    posDeprelNetwork = MultiTaskNetwork(this.posDeprelNetworkModel),
    outputErrorsInit = this.outputErrorsInit)
}
