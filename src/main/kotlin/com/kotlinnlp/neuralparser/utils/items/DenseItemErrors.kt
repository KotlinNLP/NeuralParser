/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.items

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.items.ItemErrors

/**
 * The errors object associated to a [DenseItem].
 *
 * @property array the item errors
 */
data class DenseItemErrors(val array: DenseNDArray): ItemErrors<DenseItemErrors> {

  /**
   * @return a copy of this [DenseItemErrors]
   */
  override fun copy(): DenseItemErrors = DenseItemErrors(this.array.copy())
}
