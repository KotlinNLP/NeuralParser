/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.items

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.items.ItemRelevance

/**
 * The relevance object associated to a [MultiDenseItem].
 *
 * @property arrays the item relevance
 */
data class MultiDenseItemRelevance(val arrays: List<DenseNDArray>): ItemRelevance<MultiDenseItemRelevance> {

  /**
   * @return a copy of this [MultiDenseItemRelevance]
   */
  override fun copy(): MultiDenseItemRelevance = MultiDenseItemRelevance(this.arrays.map { it.copy() })
}
