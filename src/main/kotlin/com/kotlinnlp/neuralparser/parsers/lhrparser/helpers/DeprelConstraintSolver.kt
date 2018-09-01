/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.constraints.Constraint
import com.kotlinnlp.constraints.DoubleConstraint
import com.kotlinnlp.constraints.SingleConstraint
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel

/**
 * A helper that finds the best configuration of the deprels of a dependency tree given all the possible scored deprels
 * for each token.
 *
 * @param sentence a parsing sentence
 * @param dependencyTree the dependency tree of the given sentence
 * @param constraints a list of linguistic constraints
 * @param morphoDeprelSelector a morpho-deprel selector used to build a [MorphoSyntacticToken]
 * @param scoresMap a map of valid deprels (sorted by descending score) associated to each token id
 * @param maxBeamSize the max number of parallel states that the beam supports
 * @param maxForkSize the max number of forks that can be generated from a state
 * @param maxIterations the max number of iterations of solving steps (it is the depth of beam recursion)
 */
internal class DeprelConstraintSolver(
  private val sentence: ParsingSentence,
  private val dependencyTree: DependencyTree,
  private val constraints: List<Constraint>,
  private val morphoDeprelSelector: MorphoDeprelSelector,
  scoresMap: Map<Int, List<ScoredDeprel>>,
  maxBeamSize: Int = 10,
  maxForkSize: Int = 5,
  maxIterations: Int = 10
) : BeamManager<DeprelConstraintSolver.DeprelValue, DeprelConstraintSolver.DeprelState>(
  valuesMap = scoresMap.mapValues { (_, deprels) -> deprels.map { DeprelValue(it) }.sortedByDescending { it.score } },
  maxBeamSize = maxBeamSize,
  maxForkSize = maxForkSize,
  maxIterations = maxIterations
) {

  /**
   * Raised when there is not a configuration that permits to do not violate a hard constraint.
   */
  class InvalidConfiguration : RuntimeException()

  /**
   * A value containing a deprel.
   *
   * @property deprel a scored deprel
   */
  internal data class DeprelValue(var deprel: ScoredDeprel) : Value(score = deprel.score)

  /**
   * The state that represents a configuration of deprels for the [dependencyTree].
   *
   * @param elements the list of elements in this state, sorted by diff score
   */
  internal inner class DeprelState(elements: List<StateElement<DeprelValue>>) : State(elements) {

    /**
     * The global score of the state.
     */
    override val score: Double

    /**
     * Whether this state does not violate any hard constraint.
     */
    override var isValid: Boolean = true

    /**
     * Apply the linguistic constraints to this state.
     */
    init {

      this.applyConstraints()

      // Attention: the score must be set after the constraints have been applied.
      this.score = this.elements.sumByDouble { it.value.score * dependencyTree.getAttachmentScore(it.id) }
    }

    /**
     * Apply the [constraints] to this state.
     */
    private fun applyConstraints() {

      applyDeprels(this)

      val morphoSyntacticTokens: Map<Int, MorphoSyntacticToken> =
        sentence.tokens.associate { it.id to it.toMorphoSyntactic() }

      constraints.forEach { constraint ->
        this.elements.forEach {

          val dependent: MorphoSyntacticToken = morphoSyntacticTokens.getValue(it.id)
          val governor: MorphoSyntacticToken? =
            dependencyTree.getHead(it.id)?.let { morphoSyntacticTokens.getValue(it) }

          val isVerified: Boolean = when (constraint) {
            is DoubleConstraint ->
              constraint.isVerified(dependent = dependent, governor = governor, dependencyTree = dependencyTree)
            is SingleConstraint -> constraint.isVerified(token = dependent, dependencyTree = dependencyTree)
            else -> throw RuntimeException("Invalid Constraint type: ${constraint.javaClass.simpleName}")
          }

          if (!isVerified) {
            if (constraint.isHard) this.isValid = false
            it.value.deprel = it.value.deprel.copy(score = it.value.score * constraint.penalty)
          }
        }
      }
    }

    /**
     * Convert this token into a [MorphoSyntacticToken].
     *
     * @return a new morpho-syntactic token built from this
     */
    private fun ParsingToken.toMorphoSyntactic(): MorphoSyntacticToken = this.toMutableMorphoSyntacticToken(
      dependencyRelation = DependencyRelation(
        governor = dependencyTree.getHead(this.id),
        deprel = dependencyTree.getDeprel(this.id)!!.label,
        attachmentScore = 0.0),
      morphoDeprelSelector = morphoDeprelSelector
    )
  }

  /**
   * Find the best configuration of deprels for the given [dependencyTree] that does not violates any hard constraint,
   * using the given scores map.
   *
   * @throws InvalidConfiguration if all the possible deprels configurations violate a hard constraint
   */
  fun solve() {

    this.findBestConfiguration()?.let { this.applyDeprels(it) } ?: throw InvalidConfiguration()
  }

  /**
   * Build a new state with the given elements.
   *
   * @param elements the elements that compose the building state
   *
   * @return a new state with the given elements
   */
  override fun buildState(elements: List<StateElement<DeprelValue>>): DeprelState = DeprelState(elements)

  /**
   * Set the deprels of the given state into the [dependencyTree].
   *
   * @param state a state
   */
  private fun applyDeprels(state: DeprelState) {

    state.elements.forEach {
      this.dependencyTree.setDeprel(dependent = it.id, deprel = it.value.deprel.value)
    }

    this.dependencyTree.score = state.score
  }
}
