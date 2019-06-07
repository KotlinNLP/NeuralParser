/* Copyright 2019-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers

import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.HashSet

object DirectedGraphHelper {

  /**
   * Find minimum distance incoming arcs for a given node.
   *
   * @param arcs the list of incoming arc
   *
   * @return the index of the parent node and distance.
   */
  private fun minIncomingArc(arcs: ArrayList<Pair<Int, Double>>): Pair<Int, Double> {
    val root = 0
    var min = Double.MAX_VALUE
    var argMin = root

    arcs.forEach { arc ->
      if (arc.second < min) {
        min = arc.second
        argMin = arc.first
      }
    }

    return Pair(argMin, min)
  }

  /**
   * Find directed graph made by minimum distance incoming arcs for each node.
   *
   * @param graph a [DirectedGraph]
   * @param root the root node
   *
   * @return the index of the root node.
   */
  fun minIncomingArcs(graph: DirectedGraph, root: Int): DirectedGraph {

    val pairs: ArrayList<Pair<Int, Int>> = ArrayList()
    val distances: ArrayList<Double> = ArrayList()

    graph.incomingArcs.forEachIndexed { i, _ ->
      if (i != root) {

        val arc = minIncomingArc(graph.incomingArcs[i])
        pairs.add(Pair(arc.first, i))
        distances.add(arc.second)
      }
    }
    return DirectedGraph(graphSize = graph.graphSize, pairs = pairs, distances = distances, depths = graph.depths)
  }

  /**
   * Find the root node (the lowest depth node).
   *
   * @param graph the directed graph.
   *
   * @return the index of the root node.
   */
  fun findRoot(graph: DirectedGraph): Int {
    var root = 0
    var min = Double.MAX_VALUE

    graph.depths.forEachIndexed { id, value ->
      if (value < min) {
        min = value
        root = id
      }
    }
    return root
  }

  /**
   * Performs a depth first visit of the graph
   *
   * @return the set of nodes visited
   */
  fun dfs(graph: DirectedGraph): HashSet<Int> {
    val root: Int = findRoot(graph)

    val s: Stack<Int> = Stack()
    val discovered: HashSet<Int> = HashSet()

    s.push(root)
    while (!s.empty()) {
      val vertex = s.pop()
      if (!discovered.contains(vertex)) {
        discovered.add(vertex)
        graph.adjMatrix[vertex].forEach { it -> s.push(it.first) }

      }
    }
    return discovered
  }

  /**
   * A supporting class representing node, to detect cycles
   *
   */
  class Node(val id: Int) {

    var index   = -1  // -1 is undefined
    var lowLink = -1
    var onStack = false

    override fun toString() = id.toString()
  }

  /**
   * Find all cycles in a [DirectedGraph].
   *
   * @return a list of lists (the nodes in a cycle)
   */
  fun findCycles(graph: DirectedGraph): List<List<Int>> {
    val connectedComponents: MutableList<List<Node>> = mutableListOf()
    var index = 0
    val s = Stack<Node>()
    val nodes = HashMap<Int, Node>()

    for (i in 0 until  graph.adjMatrix.size) { nodes[i] = Node(i) }

    fun strongConnect(v: Node) {
      // Set the depth index for v to the smallest unused index
      v.index = index
      v.lowLink = index
      index++
      s.push(v)
      v.onStack = true

      // consider successors of v
      for (successor in graph.adjMatrix[v.id]) {
        val successorNode: Node = nodes[successor.first]!!
        if (successorNode.index < 0) {
          // Successor w has not yet been visited; recurse on it
          strongConnect(successorNode)
          v.lowLink = minOf(v.lowLink, successorNode.lowLink)
        }
        else if (successorNode.onStack) {
          // Successor w is in stack s and hence in the current SCC
          v.lowLink = minOf(v.lowLink, successorNode.index)
        }
      }

      // If v is a root node, pop the stack and generate an SCC
      if (v.lowLink == v.index) {
        val scc = mutableListOf<Node>()
        do {
          val successorNode = s.pop()
          successorNode.onStack = false
          scc.add(successorNode)
        }
        while (successorNode != v)
        connectedComponents.add(scc)
      }
    }

    for (v in nodes.values) if (v.index < 0) strongConnect(v)
    return connectedComponents.map { it -> it.map { i -> i.id } }
  }
}