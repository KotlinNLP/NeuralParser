/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.constraints.Constraint
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.neuralparser.parsers.lhrparser.decoders.CosineDecoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.DeprelLabeler
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel

/**
 * A helper that builds the sentence dependency tree with the best configuration exploring the possible arcs and
 * the related labels with a greedy approach through a beam of parallel states.
 *
 * @param lss the latent syntactic structure of the input sentence
 * @param deprelLabeler a deprel labeler (can be null)
 * @param constraints a list of linguistic constraints (can be null)
 * @param morphoDeprelSelector a morpho-deprel selector used to build a [MorphoSyntacticToken]
 * @param deprelScoreThreshold the score threshold above which to consider a deprel valid
 */
class DependencyTreeBuilder(
  private val lss: LatentSyntacticStructure,
  private val deprelLabeler: DeprelLabeler?,
  private val constraints: List<Constraint>?,
  private val morphoDeprelSelector: MorphoDeprelSelector,
  private val deprelScoreThreshold: Double
) {

  /**
   * Find the best dependency tree that does not violates any linguistic constraint and with the highest global score.
   *
   * @return the best dependency tree built from the given LSS
   */
  fun build(): DependencyTree {

    val scores: ArcScores = CosineDecoder().decode(this.lss)

    // TODO("implement beam frobbing")

    return this.buildDependencyTree(scores)!!
  }

  /**
   * Build a new dependency tree from the latent syntactic structure [lss], using the given possible attachments.
   *
   * @param scores a map of attachments scores between pairs of tokens
   *
   * @return the annotated dependency tree with the highest score, built from the given LSS, or null if there is no
   *         valid configuration (that does not violate any hard constraint)
   */
  private fun buildDependencyTree(scores: ArcScores): DependencyTree? =
    try {
      DependencyTree(lss.sentence.tokens.map { it.id }).apply {
        assignHeads(scores)
        fixCycles(scores)
        deprelLabeler?.let { assignLabels() }
      }
    } catch (e: DeprelConstraintSolver.InvalidConfiguration) {
      null
    }

  /**
   * Assign the heads to this dependency tree using the highest scoring arcs from the given [scores].
   *
   * @param scores a map of attachments scores between pairs of tokens
   */
  private fun DependencyTree.assignHeads(scores: ArcScores) {

    val (topId: Int, topScore: Double) = scores.findHighestScoringTop()

    this.setAttachmentScore(dependent = topId, score = topScore)

    scores.keys.filter { it != topId }.forEach { depId ->

      val (govId: Int, score: Double) = scores.findHighestScoringHead(dependentId = depId, except = listOf(ArcScores.rootId))!!

      this.setArc(
        dependent = depId,
        governor = govId,
        allowCycle = true,
        score = score)
    }
  }

  /**
   * Fix possible cycles using the given [scores].
   *
   * @param scores a map of attachments scores between pairs of tokens
   */
  private fun DependencyTree.fixCycles(scores: ArcScores) = CyclesFixer(this, scores).fixCycles()

  /**
   * Annotate this dependency tree with the labels.
   */
  private fun DependencyTree.assignLabels() {

    val deprelsMap: Map<Int, List<ScoredDeprel>> = this.buildDeprelsMap()

    constraints?.let {
      DeprelConstraintSolver(
        sentence = lss.sentence,
        dependencyTree = this,
        constraints = it,
        morphoDeprelSelector = morphoDeprelSelector,
        scoresMap = deprelsMap
      ).solve()
    }
      ?: deprelsMap.forEach { tokenId, deprels -> this.setDeprel(dependent = tokenId, deprel = deprels.first().value) }
  }

  /**
   * @return a map of valid deprels (sorted by descending score) associated to each token id
   */
  private fun DependencyTree.buildDeprelsMap(): Map<Int, List<ScoredDeprel>> =

    deprelLabeler!!.predict(DeprelLabeler.Input(lss, this)).withIndex().associate { (tokenIndex, prediction) ->

      val tokenId: Int = this.elements[tokenIndex]
      val validDeprels: List<ScoredDeprel> = morphoDeprelSelector.getValidDeprels(
        deprels = prediction,
        sentence = lss.sentence,
        tokenIndex = tokenIndex,
        headIndex = this.getHead(tokenId)?.let { this.getPosition(it) })

      tokenId to validDeprels
        .filter { it.score >= deprelScoreThreshold }
        .notEmptyOr { validDeprels.subList(0, 1) }
    }

  /**
   * @param callback a callback that returns a list
   *
   * @return this list if it is not empty, otherwise the value returned by the callback
   */
  private fun <T> List<T>.notEmptyOr(callback: () -> List<T>): List<T> = if (this.isNotEmpty()) this else callback()
}
