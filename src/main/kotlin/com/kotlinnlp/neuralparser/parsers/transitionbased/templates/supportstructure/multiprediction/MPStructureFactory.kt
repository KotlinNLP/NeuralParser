/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.multiprediction

import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionModel
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionScorer
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory

/**
 * The factory of the decoding support structure for multi-prediction models.
 *
 * @param model the multi prediction scorer model used to score actions
 * @param useDropout whether to apply the dropout during the forward
 * @param propagateToInput whether to propagate the errors to the input during the backward
 */
open class MPStructureFactory(
  private val model: MultiPredictionModel,
  private val useDropout: Boolean,
  private val propagateToInput: Boolean
) : SupportStructureFactory<MPSupportStructure> {

  /**
   * Build a new [MPSupportStructure].
   *
   * @return a new multi-prediction decoding support structure
   */
  override fun globalStructure() = MPSupportStructure(
    MultiPredictionScorer(model = this.model, useDropout = this.useDropout, propagateToInput = this.propagateToInput)
  )
}
