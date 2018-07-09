/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils

import com.kotlinnlp.simplednn.core.functionalities.losses.getErrorsByHingeLoss
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The loss criterion that calculates the errors with the hinge loss method.
 */
class HingeLoss : LossCriterion {

  /**
   * @param prediction a prediction array
   * @param goldIndex the index of the gold value
   *
   * @return the errors of the given prediction
   */
  override fun getPredictionErrors(prediction: DenseNDArray, goldIndex: Int): DenseNDArray =
    getErrorsByHingeLoss(prediction = prediction, goldIndex = goldIndex)
}