/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.treeutils

/**
 * A generic element with an id and a label.
 */
interface Element<IdType: Comparable<IdType>> {

  companion object {

    /**
     * Build a new anonymous [Element].
     *
     * @param id the id
     * @param label
     *
     * @return a new element
     */
    operator fun <T: Comparable<T>>invoke(id: T, label: String) = object : Element<T> {
      override val id = id
      override val label = label
    }
  }

  /**
   * The id.
   */
  val id: IdType

  /**
   * The label.
   */
  val label: String
}
