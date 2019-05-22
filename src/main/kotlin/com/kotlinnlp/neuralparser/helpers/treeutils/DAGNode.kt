/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.treeutils

import com.kotlinnlp.utils.concat

/**
 * The Node of a directed acyclic graph.
 *
 * @property element the element
 * @property index the index
 */
class DAGNode<IdType: Comparable<IdType>>(
  val element: Element<IdType>,
  val index: Int
) {

  /**
   * The head of the node (default null).
   */
  var head: DAGNode<IdType>? = null
    set(value) { require(!this.headSet) { "Head already set." }
      field = value
      this.headSet = true
    }

  /**
   * The depth of this node in the tree.
   */
  val depth: Int by lazy { this.pathToRoot.lastIndex }

  /**
   * Contains the indexes of the node itself and its ancestors.
   */
  val pathToRoot: List<Int> by lazy {

    require(this.headSet) { "The head of the node has not yet been set."}

    listOf(this.index).concat(this.head?.pathToRoot)
  }

  /**
   * Whether the [head] has been initialized, or not.
   * Property set in the 'setHead()' method.
   */
  private var headSet: Boolean = false

  /**
   * Calculate the distance summing the length of the paths of this and the [other] nodes to the
   * Lowest Common Ancestor (LCA).
   *
   * @param other a node
   *
   * @return the distance between this and the [other] node
   */
  fun distance(other: DAGNode<IdType>): Int {

    val lca = this.pathToRoot.first { other.pathToRoot.contains(it) }

    return this.pathToRoot.indexOf(lca) + other.pathToRoot.indexOf(lca)
  }

  /**
   * @return the string representation of this node
   */
  override fun toString(): String = "($index: \"${element.label}\"${head?.let { " $it" } ?: ""})"
}
