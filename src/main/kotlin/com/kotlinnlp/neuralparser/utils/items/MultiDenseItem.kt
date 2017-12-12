/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.items

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.items.StateItem

/**
 * The atomic item that composes a [State] with multiple errors and multiple relevance dense arrays.
 *
 * @param id the unique id of this item
 */
class MultiDenseItem(override val id: Int) : StateItem<MultiDenseItem, MultiDenseItemErrors, MultiDenseItemRelevance> {

  /**
   * The errors associated to this item.
   */
  override var errors: MultiDenseItemErrors? = null

  /**
   * The relevance associated to this item.
   */
  override var relevance: MultiDenseItemRelevance? = null

  /**
   * @param errorsList the list of errors to accumulate (null errors will be ignored)
   */
  fun accumulateErrors(errorsList: List<DenseNDArray?>) {

    this.errors = MultiDenseItemErrors(errorsList.mapIndexed { index, errors ->
      if (this.errors == null || errors == null) {
        errors?.copy()
      } else {
        if (this.errors!!.arrays[index] == null) {
          errors.copy()
        } else {
          this.errors!!.arrays[index]!!.assignSum(errors)
        }
      }
    })
  }

  /**
   * @return a copy of this [MultiDenseItem]
   */
  override fun copy(): MultiDenseItem {

    val clone = MultiDenseItem(id = this.id)

    if (clone.errors != null) {
      clone.errors = this.errors!!.copy()
    }

    if (clone.relevance != null) {
      clone.relevance = this.relevance!!.copy()
    }

    return clone
  }
}
