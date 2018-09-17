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
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel

/**
 * A helper that finds the best configuration of the labels of a dependency tree given the probability distribution of
 * the grammatical configurations for each token.
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
internal class ConstraintsSolver(
  private val sentence: ParsingSentence,
  private val dependencyTree: DependencyTree,
  private val constraints: List<Constraint>,
  private val morphoDeprelSelector: MorphoDeprelSelector,
  scoresMap: Map<Int, List<ScoredDeprel>>,
  maxBeamSize: Int = 5,
  maxForkSize: Int = 3,
  maxIterations: Int = 10
) : BeamManager<ConstraintsSolver.GrammarValue, ConstraintsSolver.GrammarState>(
  valuesMap = scoresMap.mapValues { (_, grammar) -> grammar.map { GrammarValue(it) }.sortedByDescending { it.score } },
  maxBeamSize = maxBeamSize,
  maxForkSize = maxForkSize,
  maxIterations = maxIterations
) {

  /**
   * Raised when there is not a configuration that permits to do not violate a hard constraint.
   */
  class InvalidConfiguration : RuntimeException()

  /**
   * A value containing a grammatical configuration.
   *
   * @property grammar a scored grammatical configuration
   */
  internal data class GrammarValue(var grammar: ScoredDeprel) : Value() {

    /**
     * The score of this value.
     */
    override var score: Double = this.grammar.score
  }

  /**
   * The state that represents a grammatical configuration of the [dependencyTree].
   *
   * @param elements the list of elements in this state, sorted by diff score
   */
  internal inner class GrammarState(elements: List<StateElement<GrammarValue>>) : State(elements) {

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

      applyConfiguration(this)

      val tokens: List<MorphoSyntacticToken> = sentence.tokens.map { it.toMorphoSyntactic() }
      val tokensMap: Map<Int, MorphoSyntacticToken> = tokens.associateBy { it.id }

      constraints.forEach { constraint ->
        this.elements.forEach {

          val isVerified: Boolean =
            constraint.isVerified(token = tokensMap.getValue(it.id), tokens = tokens, dependencyTree = dependencyTree)

          if (!isVerified) {
            if (constraint.isHard) this.isValid = false
            it.value.grammar = it.value.grammar.copy(score = it.value.score * constraint.penalty)
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
      syntacticRelation = SyntacticRelation(
        governor = dependencyTree.getHead(this.id),
        grammaticalConfiguration = dependencyTree.getGrammaticalConfiguration(this.id)!!,
        attachmentScore = 0.0),
      morphoDeprelSelector = morphoDeprelSelector
    )
  }

  /**
   * Find the best grammatical configuration for the given [dependencyTree] that does not violates any hard constraint,
   * using the given scores map.
   *
   * @throws InvalidConfiguration if all the possible configurations violate a hard constraint
   */
  fun solve() {

    this.findBestConfiguration()?.let { this.applyConfiguration(it) } ?: throw InvalidConfiguration()
  }

  /**
   * Build a new state with the given elements.
   *
   * @param elements the elements that compose the building state
   *
   * @return a new state with the given elements
   */
  override fun buildState(elements: List<StateElement<GrammarValue>>): GrammarState = GrammarState(elements)

  /**
   * Set the grammatical configuration of the given state into the [dependencyTree].
   *
   * @param state a state
   */
  private fun applyConfiguration(state: GrammarState) {

    state.elements.forEach {
      this.dependencyTree.setGrammaticalConfiguration(dependent = it.id, deprel = it.value.grammar.value)
    }

    this.dependencyTree.score = state.score
  }
}
