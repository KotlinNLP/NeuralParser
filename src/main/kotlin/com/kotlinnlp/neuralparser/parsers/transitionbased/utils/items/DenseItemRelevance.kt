/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.items.ItemRelevance

/**
 * The relevance object associated to a [DenseItem].
 *
 * @property array the item relevance
 */
data class DenseItemRelevance(val array: DenseNDArray): ItemRelevance<DenseItemRelevance> {

  /**
   * @return a copy of this [DenseItemRelevance]
   */
  override fun copy(): DenseItemRelevance = DenseItemRelevance(this.array.copy())
}
