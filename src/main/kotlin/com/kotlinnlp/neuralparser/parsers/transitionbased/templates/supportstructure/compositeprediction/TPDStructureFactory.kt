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
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory

/**
 * The factory of the decoding support structure for composite TPD (Transition + POS tag + Deprel) prediction models.
 *
 * @param transitionNetwork the neural network used to score the transitions
 * @param posNetwork the neural network used to score the POS tags
 * @param deprelNetwork the neural network used to score the deprels
 * @param outputErrorsInit the default initialization of the output errors
 */
open class TPDStructureFactory(
  private val transitionNetwork: NeuralNetwork,
  private val posNetwork: NeuralNetwork,
  private val deprelNetwork: NeuralNetwork,
  private val outputErrorsInit: OutputErrorsInit
) : SupportStructureFactory<TPDSupportStructure> {

  /**
   * Build a new [TPDSupportStructure].
   *
   * @return a new single-prediction decoding support structure
   */
  override fun globalStructure() = TPDSupportStructure(
    transitionProcessor = FeedforwardNeuralProcessor(
      neuralNetwork = this.transitionNetwork,
      useDropout = false, // TODO: enable during training
      propagateToInput = true),
    posProcessor = FeedforwardNeuralProcessor(
      neuralNetwork = this.posNetwork,
      useDropout = false, // TODO: enable during training
      propagateToInput = true),
    deprelProcessor = FeedforwardNeuralProcessor(
      neuralNetwork = this.deprelNetwork,
      useDropout = false, // TODO: enable during training
      propagateToInput = true),
    outputErrorsInit = this.outputErrorsInit)
}
