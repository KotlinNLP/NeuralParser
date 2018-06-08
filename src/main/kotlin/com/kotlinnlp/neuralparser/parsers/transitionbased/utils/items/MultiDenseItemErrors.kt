/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.items.ItemErrors

/**
 * The errors object associated to a [MultiDenseItem].
 *
 * @property arrays the item errors
 */
data class MultiDenseItemErrors(val arrays: List<DenseNDArray?>): ItemErrors<MultiDenseItemErrors> {

  /**
   * @return a copy of this [MultiDenseItemErrors]
   */
  override fun copy(): MultiDenseItemErrors = MultiDenseItemErrors(this.arrays.map { it?.copy() })
}
