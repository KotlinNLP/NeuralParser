/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.constraints.Constraint
import com.kotlinnlp.dependencytree.CycleDetectedError
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.selector.LabelerSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.Labeler
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar
import com.kotlinnlp.neuralparser.utils.notEmptyOr

/**
 * A helper that builds the sentence dependency tree with the best configuration exploring the possible arcs and
 * the related labels with a greedy approach through a beam of parallel states.
 *
 * @param lss the latent syntactic structure of the input sentence
 * @param labeler a classifier of grammatical configurations (can be null)
 * @param constraints a list of linguistic constraints (can be null)
 * @param labelerSelector a labeler selector used to build a [MorphoSynToken]
 * @param labelerScoreThreshold the score threshold above which to consider a labeler output valid
 * @param maxBeamSize the max number of parallel states that the beam supports
 * @param maxForkSize the max number of forks that can be generated from a state
 * @param maxIterations the max number of iterations of solving steps (it is the depth of beam recursion)
 */
internal class DependencyTreeBuilder(
  private val lss: LatentSyntacticStructure,
  private val scoresMap: ArcScores,
  private val labeler: Labeler?,
  private val constraints: List<Constraint>?,
  private val labelerSelector: LabelerSelector,
  private val labelerScoreThreshold: Double,
  maxBeamSize: Int = 5,
  maxForkSize: Int = 3,
  maxIterations: Int = 10
) : BeamManager<DependencyTreeBuilder.ArcValue, DependencyTreeBuilder.TreeState>(
  valuesMap = lss.sentence.tokens.associate {
    val sortedArcs: List<ArcScores.Arc> = scoresMap.getSortedArcs(it.id)
    val threshold: Double = 1.0 / sortedArcs.size // it is the distribution mean
    it.id to sortedArcs.filter { it.score >= threshold }.notEmptyOr { sortedArcs }.map { arc ->
      ArcValue(dependentId = it.id, governorId = arc.governorId, score = arc.score)
    }
  },
  maxBeamSize = maxBeamSize,
  maxForkSize = maxForkSize,
  maxIterations = maxIterations
) {

  /**
   * An arc of a state, including information about its dependent token and its index in the related scored arcs list.
   *
   * @property dependentId the id of the dependent of this arc
   * @property governorId the id of the governor of this arc
   * @property score the arc score
   */
  internal data class ArcValue(val dependentId: Int, val governorId: Int, override var score: Double) : Value()

  /**
   * The state that contains the configuration of a possible [DependencyTree].
   *
   * @param elements the list of elements in this state, sorted by diff score
   */
  internal inner class TreeState(elements: List<StateElement<ArcValue>>) : State(elements) {

    /**
     * The global score of the state.
     */
    override val score: Double get() = this.tree.score

    /**
     * The dependency tree of this state.
     */
    val tree = DependencyTree(lss.sentence.tokens.map { it.id })

    /**
     * Whether the [tree] is a fully-connected DAG (Direct Acyclic Graph).
     */
    override var isValid: Boolean = true

    /**
     * Initialize the dependency tree.
     */
    init {

      try {

        elements.forEach {
          if (it.value.governorId > -1)
            this.tree.setArc(dependent = it.value.dependentId, governor = it.value.governorId, score = it.value.score)
          else
            this.tree.setAttachmentScore(dependent = it.value.dependentId, score = it.value.score) // it is the top
        }

        if (this.tree.hasSingleRoot())
          labeler?.let { this.tree.assignLabels() }
        else
          this.isValid = false

      } catch (e: CycleDetectedError) {
        this.isValid = false
      }
    }
  }

  /**
   * Find the best dependency tree that does not violates any linguistic constraint and with the highest global score.
   *
   * @return the best dependency tree built from the given LSS
   */
  fun build(): DependencyTree = this.findBestConfiguration()?.tree ?: this.buildDependencyTree()

  /**
   * Build a new state with the given elements.
   *
   * @param elements the elements that compose the building state
   *
   * @return a new state with the given elements
   */
  override fun buildState(elements: List<StateElement<ArcValue>>): TreeState = TreeState(elements)

  /**
   * Build a new dependency tree from the latent syntactic structure [lss], using the possible attachments in the
   * [scoresMap].
   *
   * @return the annotated dependency tree with the highest score, built from the given LSS, or null if there is no
   *         valid configuration (that does not violate any hard constraint)
   */
  private fun buildDependencyTree(): DependencyTree =
    DependencyTree(lss.sentence.tokens.map { it.id }).apply {
      assignBestHeads()
      fixCycles()
      labeler?.let { assignLabels() }
    }

  /**
   * Assign the heads to this dependency tree using the highest scoring arcs of the [scoresMap].
   */
  private fun DependencyTree.assignBestHeads() {

    val (topId: Int, topScore: Double) = scoresMap.findHighestScoringTop()

    this.setAttachmentScore(dependent = topId, score = topScore)

    this.elements.filter { it != topId }.forEach { depId ->

      val (govId: Int, score: Double) =
        scoresMap.findHighestScoringHead(dependentId = depId, except = listOf(ArcScores.rootId))!!

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
  private fun DependencyTree.fixCycles() = CyclesFixer(dependencyTree = this, arcScores = scoresMap).fixCycles()

  /**
   * Annotate this dependency tree with the labels.
   */
  private fun DependencyTree.assignLabels() {

    val configMap: Map<Int, List<ScoredGrammar>> = this.buildConfigurationsMap()

    if (constraints != null)
      ConstraintsSolver(
        sentence = lss.sentence,
        dependencyTree = this,
        constraints = constraints,
        labelerSelector = labelerSelector,
        scoresMap = configMap
      ).solve()
    else
      configMap.forEach { tokenId, configurations ->
        this.setGrammaticalConfiguration(dependent = tokenId, configuration = configurations.first().config)
      }
  }

  /**
   * @return a map of valid grammatical configurations (sorted by descending score) associated to each token id
   */
  private fun DependencyTree.buildConfigurationsMap(): Map<Int, List<ScoredGrammar>> =

    labeler!!.predict(Labeler.Input(lss, this)).withIndex().associate { (tokenIndex, configurations) ->

      val tokenId: Int = this.elements[tokenIndex]
      val validConfigurations: List<ScoredGrammar> = labelerSelector.getValidConfigurations(
        configurations = configurations,
        sentence = lss.sentence,
        tokenIndex = tokenIndex,
        headIndex = this.getHead(tokenId)?.let { this.getPosition(it) })

      tokenId to validConfigurations
        .filter { it.score >= labelerScoreThreshold }
        .notEmptyOr { validConfigurations.subList(0, 1) }
    }
}
