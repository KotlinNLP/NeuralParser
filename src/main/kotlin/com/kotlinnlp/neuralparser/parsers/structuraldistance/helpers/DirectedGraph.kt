/* Copyright 2019-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers

/**
 * A directed graph
 *
 * @param graphSize The number of nodes in the graph.
 * @param pairs The list of nodes pairs.
 * @param distances The list of distances between corresponding nodes pairs.
 * @param depths The list of nodes depths.
 * @param penalty If true, set a backward edge between nodes at the same depths (threshold 1.0)
 */
class DirectedGraph (val graphSize: Int,
                     val pairs: List<Pair<Int, Int>>,
                     val distances: List<Double>,
                     val depths: List<Double>,
                     val penalty: Boolean = false
) {

  val adjMatrix: Array<ArrayList<Pair<Int, Double>>> = Array(depths.size) { ArrayList<Pair<Int, Double>>() }

  val incomingArcs: Array<ArrayList<Pair<Int, Double>>> = Array(depths.size) { ArrayList<Pair<Int, Double>>() }

  /**
   * Set the adjacency matrix. For each node in the list, add the neighbours (outcoming) nodes.
   *
   * @param pair A node pair, unordered.
   * @param distance The distance between nodes.
   */
  private fun setAdjacencyMatrix(pair: Pair<Int, Int>, distance: Double){

    if (depths[pair.first] > depths[pair.second]) {
      adjMatrix[pair.second].add(Pair(pair.first, distance))
    } else {
      adjMatrix[pair.first].add(Pair(pair.second, distance))
    }

    if (depths[pair.first] > depths[pair.second]) {
      incomingArcs[pair.first].add(Pair(pair.second, distance))
    } else {
      incomingArcs[pair.second].add(Pair(pair.first, distance))
    }

  }

  /**
   * Set the adjacency matrix.
   */
  init {
    if(penalty) {
      this.pairs.zip(distances) { pair, distance ->

        val distanceDiff = Math.abs((depths[pair.first] - depths[pair.second]))
        if (distanceDiff > 1.0) {
          this.setAdjacencyMatrix(pair, distance)
        }

        if (distanceDiff <= 1.0) {
          adjMatrix[pair.second].add(Pair(pair.first, distance))
          adjMatrix[pair.first].add(Pair(pair.second, distance))
          incomingArcs[pair.first].add(Pair(pair.second, distance))
          incomingArcs[pair.second].add(Pair(pair.first, distance))
        }
      }

    }else{
      this.pairs.zip(distances) { pair, distance ->

        this.setAdjacencyMatrix(pair, distance)
      }
    }
  }

  /**
   * Convert the graph to triples (parent, child, distance)
   *
   * @rteurn a list of triples (governor, dependent, distance). Ids start from 1 to N
   */
  fun toTriples():List<Triple<Int, Int, Double>> {

    val tree = ArrayList(emptyList<Triple<Int, Int, Double>>())

    adjMatrix.forEachIndexed { i, it -> it.forEach{ tree.add(Triple(i + 1, it.first + 1, it.second)) } }

    return tree
  }
}