/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The LossCriterion interface.
 */
interface LossCriterion {

  companion object {

    /**
     * The LossCriterion builder.
     *
     * @param type the loss criterion type
     */
    operator fun invoke(type: LossCriterionType): LossCriterion = when (type) {
      LossCriterionType.Softmax -> Softmax()
      LossCriterionType.HingeLoss -> HingeLoss()
    }
  }

  /**
   * @param prediction a prediction array
   * @param goldIndex the index of the gold value
   *
   * @return the errors of the given prediction
   */
  fun getPredictionErrors(prediction: DenseNDArray, goldIndex: Int): DenseNDArray
}
