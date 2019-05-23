/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.treeutils

/**
 * The [DAGNode]s factory.
 *
 * @param elements a list of elements
 * @param arcs a list of arcs
 */
class DAGNodesFactory<IdType: Comparable<IdType>>(
  private val elements: List<Element<IdType>>,
  private val arcs: List<Pair<IdType, IdType?>>
) {

  /**
   * Map the Ids with the linear indexes of the elements in the list.
   */
  private val idsToIndexes: Map<IdType, Int> = this.elements.foldIndexed(mutableMapOf()) { index, acc, token ->
    acc[token.id] = index
    acc
  }

  /**
   * Build a new list of nodes.
   *
   * @return a new list of nodes
   */
  fun newNodes(): List<DAGNode<IdType>> = this.elements.mapIndexed { index, token -> DAGNode(token, index) }.also { nodes ->

    this.arcs.forEach { (depId, govId) ->

      assignHead(depId, govId, nodes)
    }
  }

  /**
   * Assign the heads into the nodes.
   *
   * @param depId the dependent id
   * @param govId the governor id (can be null in case of root)
   * @param nodes the list of nodes
   */
  private fun assignHead(depId: IdType, govId: IdType?, nodes: List<DAGNode<IdType>>) {

    val depNodeIndex: Int = this.idsToIndexes.getValue(depId)

    if (govId == null)
      nodes[depNodeIndex].assignHead(null) // root
    else
      this.idsToIndexes.getValue(govId).let { govNodeIndex ->
        nodes[depNodeIndex].assignHead(nodes[govNodeIndex])
      }
  }
}
