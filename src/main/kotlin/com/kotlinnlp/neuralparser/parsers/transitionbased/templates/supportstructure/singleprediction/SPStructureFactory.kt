/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.singleprediction

import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory

/**
 * The factory of the decoding support structure for single prediction models.
 *
 * @param network the neural network used to score actions
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 * @param outputErrorsInit the default initialization of the [network] processor output errors
 */
class SPStructureFactory(
  val network: NeuralNetwork,
  private val useDropout: Boolean,
  private val propagateToInput: Boolean,
  val outputErrorsInit: OutputErrorsInit
) : SupportStructureFactory<SPSupportStructure> {

  /**
   * Build a new [SPSupportStructure].
   *
   * @return a new single-prediction decoding support structure
   */
  override fun globalStructure() = SPSimpleSupportStructure(
    processor = FeedforwardNeuralProcessor(
      neuralNetwork = this.network,
      useDropout = this.useDropout,
      propagateToInput = this.propagateToInput),
    outputErrorsInit = this.outputErrorsInit)
}
