/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.items.StateItem

/**
 * The atomic item that composes a [State] with errors and relevance as dense arrays.
 *
 * @param id the unique id of this item
 */
class DenseItem(override val id: Int) : StateItem<DenseItem, DenseItemErrors, DenseItemRelevance> {

  /**
   * The errors associated to this item.
   */
  override var errors: DenseItemErrors? = null

  /**
   * The relevance associated to this item.
   */
  override var relevance: DenseItemRelevance? = null

  /**
   * @param errors the errors array to accumulate
   */
  fun accumulateErrors(errors: DenseNDArray) {

    if (this.errors != null) {
      this.errors!!.array.assignSum(errors)
    } else {
      this.errors = DenseItemErrors(array = errors.copy())
    }
  }

  /**
   * @return a copy of this [DenseItem]
   */
  override fun copy(): DenseItem {

    val clone = DenseItem(id = this.id)

    if (clone.errors != null) {
      clone.errors = this.errors!!.copy()
    }

    if (clone.relevance != null) {
      clone.relevance = this.relevance!!.copy()
    }

    return clone
  }
}
