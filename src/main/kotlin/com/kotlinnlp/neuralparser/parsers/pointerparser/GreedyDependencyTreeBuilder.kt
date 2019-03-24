/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.pointerparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.language.ParsingSentence

/**
 * TODO: uniform with com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.GreedyDependencyTreeBuilder
 *
 * A helper that builds the dependency tree with the highest scoring configurations.
 *
 * @param sentence the input sentence
 * @param scoresMap the scores map
 */
internal class GreedyDependencyTreeBuilder(
  private val sentence: ParsingSentence,
  private val scoresMap: ScoredArcs) {

  /**
   * Build a new dependency tree using the possible attachments in the [scoresMap].
   *
   * @return the annotated dependency tree with the highest local scores
   */
  fun build() = DependencyTree(this.sentence.tokens.map { it.id }).apply {
    assignHighestScoringHeads()
    fixCycles()
  }

  /**
   * Assign the heads to this dependency tree using the highest scoring arcs of the [scoresMap].
   */
  private fun DependencyTree.assignHighestScoringHeads() {

    /**
     * @return the highest scoring element that points to the root
     */
    val topId: Int = scoresMap.findHighestScoringTop()

    this.setAttachmentScore(dependent = topId, score = 0.0)

    this.elements.filter { it != topId }.forEach { depId ->

      val (govId: Int, score: Double) = scoresMap.findHighestScoringHead(
        dependentId = depId,
        except = listOf(ScoredArcs.rootId))!!

      this.setArc(
        dependent = depId,
        governor = govId,
        allowCycle = true,
        score = score)
    }
  }

  /**
   * Fix possible cycles using the [scoresMap].
   */
  private fun DependencyTree.fixCycles() = CyclesFixer(dependencyTree = this, scoredArcs = scoresMap).fixCycles()
}
