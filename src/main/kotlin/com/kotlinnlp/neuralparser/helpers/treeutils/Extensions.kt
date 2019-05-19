/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.treeutils

import com.kotlinnlp.utils.combine

/**
 * Print the nodes.
 */
fun List<DAGNode<*>>.printNodes() { this.forEach { println(it) } }

/**
 * Print the nodes depth.
 */
fun List<DAGNode<*>>.printNodesDepth() {

  this.forEach { println("${it.index}: \"${it.element.label}\" >> ${it.depth}") }
}

/**
 * Print the distance of all pairs of node.
 */
fun <T: Comparable<T>>List<DAGNode<T>>.printAllPairsDistance() {

  this.combine().forEach { (first, second) ->

    println("\"${first.element.label}\" - \"${second.element.label}\" >> ${first.distance(second)}")
  }
}