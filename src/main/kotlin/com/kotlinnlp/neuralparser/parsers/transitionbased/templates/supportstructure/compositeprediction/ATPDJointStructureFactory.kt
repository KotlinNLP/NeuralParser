/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory

/**
 * The factory of the decoding support structure for joint composite ATPD (Action + Transition + joint POS tag and
 * Deprel) prediction models.
 *
 * @param appliedActionsNetwork the recurrent neural network used to encode the applied actions
 * @param transitionNetwork the neural network used to score the transitions
 * @param posDeprelNetworkModel the neural model of the multi-task network used to score POS tags and deprels
 * @param outputErrorsInit the default initialization of the output errors
 */
open class ATPDJointStructureFactory(
  private val appliedActionsNetwork: NeuralNetwork,
  private val transitionNetwork: NeuralNetwork,
  private val posDeprelNetworkModel: MultiTaskNetworkModel,
  private val outputErrorsInit: OutputErrorsInit
) : SupportStructureFactory<ATPDJointSupportStructure> {

  /**
   * Build a new [ATPDJointSupportStructure].
   *
   * @return a new ATPD Joint decoding support structure
   */
  override fun globalStructure() = ATPDJointSupportStructure(
    transitionProcessor = FeedforwardNeuralProcessor(this.transitionNetwork),
    posDeprelNetwork = MultiTaskNetwork(this.posDeprelNetworkModel),
    actionProcessor = RecurrentNeuralProcessor(this.appliedActionsNetwork),
    outputErrorsInit = this.outputErrorsInit)
}
