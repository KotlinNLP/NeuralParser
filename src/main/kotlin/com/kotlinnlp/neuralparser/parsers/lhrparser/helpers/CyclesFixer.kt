/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree

/**
 * Naive strategy to fix possible cycles in a [dependencyTree].
 *
 * @param dependencyTree the dependency tree
 * @param arcScores the scores of the arcs between pair of tree elements
 */
internal class CyclesFixer(private val dependencyTree: DependencyTree, private val arcScores: ArcScores) {

  /**
   * The set of direct elements of the tree (elements that aren't involved in cycles).
   */
  private lateinit var directElements: Set<Int>

  /**
   * Fix the cycles of the dependency tree.
   */
  fun fixCycles() {

    val cycles: List<DependencyTree.Path> = this.dependencyTree.getCycles()

    this.directElements = this.dependencyTree.elements.toSet() - cycles.toElementsSet()

    cycles.forEach { this.fixCycle(it) }
  }

  /**
   * @return the set of elements from a list of path
   */
  private fun List<DependencyTree.Path>.toElementsSet(): Set<Int> {

    val elements = mutableSetOf<Int>()
    this.forEach { path -> elements += path.arcs.map { it.dependent } }
    return elements
  }

  /**
   * Remove a [cycle] from the dependency tree.
   *
   * @param cycle a cycle of the dependency tree
   */
  private fun fixCycle(cycle: DependencyTree.Path) {

    val dep: Int = this.removeLowestScoringArc(cycle.arcs)
    val (newGov: Int, score: Double) = this.findBestGovernor(dep)
    this.dependencyTree.setArc(dependent = dep, governor = newGov, score = score)
  }

  /**
   * Remove the lowest scoring arc and return the related dependent to be reattached.
   *
   * @param arcs a list of arcs
   *
   * @return the element to be reattached.
   */
  private fun removeLowestScoringArc(arcs: List<DependencyTree.Arc>): Int {

    val arc: DependencyTree.Arc = this.getLowestScoringArc(arcs)
    this.dependencyTree.removeArc(dependent = arc.dependent, governor = arc.governor)
    return arc.dependent
  }

  /**
   * @param arcs a list of arcs
   *
   * @return the lowest scoring arc according to the [arcScores].
   */
  private fun getLowestScoringArc(arcs: List<DependencyTree.Arc>): DependencyTree.Arc =
    arcs.minBy { arc -> this.arcScores.getScore(dependentId = arc.dependent, governorId = arc.governor) }!!

  /**
   * Find the best governor for the given element that doesn't introduce a cycle.
   *
   * @param element an element of the dependency tree
   *
   * @return the new governor id and the related score
   */
  private fun findBestGovernor(element: Int): Pair<Int, Double> {

    val headsMap: Map<Int, Double> = this.arcScores.getHeadsMap(element)

    val candidates: List<Int> = this.directElements.intersect(headsMap.keys).filter { candidateGov ->
      !this.dependencyTree.introduceCycle(dependent = element, governor = candidateGov)
    }

    return headsMap.filter { it.key in candidates }.maxBy { it.value }!!.toPair()
  }
}
