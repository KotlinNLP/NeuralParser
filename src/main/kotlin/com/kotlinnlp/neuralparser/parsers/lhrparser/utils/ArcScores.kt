/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.utils

/**
 * The structure that contains the scored arcs.
 * The score is a double value.
 * Scores are mapped by dependents to governors ids ().
 *
 * @param scores the scores map
 */
class ArcScores(
  private val scores: Map<Int, Map<Int, Double>>
) : Map<Int, Map<Int, Double>> by scores {

  companion object {

    /**
     * The root is intended to have id = -1
     */
    const val rootId: Int = -1
  }

  /**
   * @param dependentId the dependent id
   * @param governorId the governor id (can be null)
   *
   * @return the attachment score between the given [dependentId] and [governorId]
   */
  fun getScore(dependentId: Int, governorId: Int?): Double =
    this.scores.getValue(dependentId).getValue(governorId ?: rootId)

  /**
   * Find the best head of a given dependent.
   *
   * @param dependentId the id of the dependent token
   * @param except list of heads not to be considered
   *
   * @return the highest scoring head-score entry
   */
  fun findHighestScoringHead(dependentId: Int, except: List<Int> = emptyList()): Pair<Int, Double>? =
    this.scores
      .getValue(dependentId)
      .filterNot { it.key in except }
      .maxBy { it.value }
      ?.toPair()

  /**
   * @return the highest scoring element that points to the root
   */
  fun findHighestScoringTop(): Pair<Int, Double> {

    val topId = this.maxBy { it.value.getValue(rootId) }!!.key

    return Pair(topId, this.scores.getValue(topId).getValue(rootId))
  }
}
