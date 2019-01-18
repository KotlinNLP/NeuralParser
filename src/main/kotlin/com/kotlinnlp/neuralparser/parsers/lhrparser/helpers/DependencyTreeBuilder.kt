/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.dependencytree.CycleDetectedError
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.lssencoder.LatentSyntacticStructure
import com.kotlinnlp.lssencoder.decoder.ScoredArcs
import com.kotlinnlp.lssencoder.decoder.ScoredArcs.Companion.rootId
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.morphopercolator.MorphoPercolator
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.Labeler
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar
import com.kotlinnlp.utils.BeamManager
import com.kotlinnlp.utils.notEmptyOr

/**
 * A helper that builds the sentence dependency tree with the best configuration exploring the possible arcs and
 * the related labels with a greedy approach through a beam of parallel states.
 *
 * @param lss the latent syntactic structure of the input sentence
 * @param labeler a classifier of grammatical configurations (can be null)
 * @param constraints a list of linguistic constraints (can be null)
 * @param morphoPercolator a percolator of morphology properties
 * @param maxBeamSize the max number of parallel states that the beam supports (-1 = infinite)
 * @param maxForkSize the max number of forks that can be generated from a state (-1 = infinite)
 * @param maxIterations the max number of iterations of solving steps (it is the depth of beam recursion, -1 = infinite)
 */
internal class DependencyTreeBuilder(
  private val lss: LatentSyntacticStructure<ParsingToken, ParsingSentence>,
  scoredArcs: ScoredArcs,
  private val labeler: Labeler?,
  private val constraints: List<Constraint>?,
  private val morphoPercolator: MorphoPercolator,
  maxBeamSize: Int = 5,
  maxForkSize: Int = 3,
  maxIterations: Int = 10
) : BeamManager<DependencyTreeBuilder.ArcValue, DependencyTreeBuilder.TreeState>(
  valuesMap = getValuesMap(tokensIds = lss.sentence.tokens.map { it.id }, scoresMap = scoredArcs),
  maxBeamSize = maxBeamSize,
  maxForkSize = maxForkSize,
  maxIterations = maxIterations
) {

  companion object {

    /**
     * @param tokensIds list of tokens ids
     * @param scoresMap the scored arcs
     *
     * @return the map of element ids associated to their possible values, sorted by descending score
     */
    fun getValuesMap(tokensIds: List<Int>, scoresMap: ScoredArcs): Map<Int, List<ArcValue>> {

      return tokensIds.associate { tokenId ->

        val sortedArcs: List<ScoredArcs.Arc> = scoresMap.getSortedArcs(tokenId)
        val threshold: Double = 1.0 / sortedArcs.size // it is the distribution mean

        tokenId to sortedArcs.filter { it.score >= threshold }.notEmptyOr { sortedArcs }.map { arc ->
          ArcValue(dependentId = tokenId, governorId = arc.governorId, score = arc.score)
        }
      }
    }
  }

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
          if (it.value.governorId > rootId)
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
  fun build(): DependencyTree? = this.findBestConfiguration()?.tree

  /**
   * Build a new state with the given elements.
   *
   * @param elements the elements that compose the building state
   *
   * @return a new state with the given elements
   */
  override fun buildState(elements: List<StateElement<ArcValue>>): TreeState = TreeState(elements)

  /**
   * Annotate this dependency tree with the labels.
   */
  private fun DependencyTree.assignLabels() {

    val configMap: Map<Int, List<ScoredGrammar>> = labeler!!.predict(Labeler.Input(lss, this))

    if (constraints != null)
      LabelsSolver(
        sentence = lss.sentence,
        dependencyTree = this,
        constraints = constraints,
        morphoPercolator = morphoPercolator,
        scoresMap = configMap
      ).solve()
    else
      configMap.forEach { tokenId, configurations ->
        this.setGrammaticalConfiguration(dependent = tokenId, configuration = configurations.first().config)
      }
  }
}
