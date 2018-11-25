/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.lssencoder.LatentSyntacticStructure
import com.kotlinnlp.lssencoder.decoder.ScoredArcs
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.Labeler

/**
 * A helper that builds the dependency tree with the highest scoring configurations.
 *
 * @param lss the latent syntactic structure of the input sentence
 */
internal class GreedyDependencyTreeBuilder(
  private val lss: LatentSyntacticStructure<ParsingToken, ParsingSentence>,
  private val scoresMap: ScoredArcs,
  private val labeler: Labeler?
) {

  /**
   * Build a new dependency tree from the latent syntactic structure [lss], using the possible attachments in the
   * [scoresMap].
   *
   * @return the annotated dependency tree with the highest score, built from the given LSS
   */
  fun build() = DependencyTree(this.lss.sentence.tokens.map { it.id }).apply {
    assignHighestScoringHeads()
    fixCycles()
    assignLabels()
  }

  /**
   * Assign the heads to this dependency tree using the highest scoring arcs of the [scoresMap].
   */
  private fun DependencyTree.assignHighestScoringHeads() {

    val (topId: Int, topScore: Double) = scoresMap.findHighestScoringTop()

    this.setAttachmentScore(dependent = topId, score = topScore)

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

  /**
   * Annotate this dependency tree with the labels.
   */
  private fun DependencyTree.assignLabels() {

    labeler!!.predict(Labeler.Input(lss, this)).forEach { tokenId, configurations ->
      this.setGrammaticalConfiguration(dependent = tokenId, configuration = configurations.first().config)
    }
  }
}
