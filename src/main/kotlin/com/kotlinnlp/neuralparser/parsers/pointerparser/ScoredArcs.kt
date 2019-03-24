/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.pointerparser

/**
 * TODO: uniform with the ScoredArcs of the LSSEncoder.
 * TODO: for example, implementing multiple strategies to find the 'root'.
 *
 * The score is a double in the range [0, 1].
 * Scores are mapped by dependents to governors ids.
 *
 * @param scores the scores map
 */
@Suppress("UNUSED")
class ScoredArcs(private val scores: Map<Int, Map<Int, Double>>) {

  companion object {

    /**
     * The root is intended to have id = -1.
     */
    const val rootId: Int = -1
  }

  /**
   * The arc related to a specific dependent.
   *
   * @property governorId the id of the governor
   * @property score the arc score
   */
  data class Arc(val governorId: Int, val score: Double)

  /**
   * Arcs sorted by descending score and associated by dependent id.
   */
  private val sortedArcs: Map<Int, List<Arc>> by lazy {
    this.scores.mapValues { (_, arcs) ->
      arcs.map { (govId, score) -> Arc(governorId = govId, score = score) }.sortedByDescending { it.score }
    }
  }

  /**
   * @param dependentId the dependent id
   * @param governorId the governor id (can be null)
   *
   * @return the arc score between the given [dependentId] and [governorId]
   */
  fun getScore(dependentId: Int, governorId: Int?): Double =
    this.scores.getValue(dependentId).getValue(governorId ?: rootId)

  /**
   * @param dependentId the dependent id
   *
   * @return the map of head scores of the given [dependentId] associated by each possible governor
   */
  fun getHeadsMap(dependentId: Int): Map<Int, Double> = this.scores.getValue(dependentId)

  /**
   * @param dependentId the dependent id
   *
   * @return the list of possible arcs of the given [dependentId], sorted by diff score
   */
  fun getSortedArcs(dependentId: Int): List<Arc> = this.sortedArcs.getValue(dependentId)

  /**
   * Find the best head of a given dependent.
   *
   * @param dependentId the id of the dependent token
   * @param except list of heads not to be considered
   *
   * @return the highest scoring head-score entry
   */
  fun findHighestScoringHead(dependentId: Int, except: List<Int> = emptyList()): Pair<Int, Double>? =
    this.getHeadsMap(dependentId)
      .filterNot { it.key in except }
      .maxBy { it.value }
      ?.toPair()

  /**
   * Find the best head of a given dependent.
   *
   * @param dependentId the id of the dependent token
   * @param except the head not to be considered
   *
   * @return the highest scoring head-score entry
   */
  fun findHighestScoringHead(dependentId: Int, except: Int) = this.findHighestScoringHead(dependentId, listOf(except))

  /**
   * @return the element that points to the root
   */
  fun findHighestScoringTop(): Int = this.scores.minBy { it.value.maxBy { head -> head.value }?.value ?: 1.0 }!!.key
}
