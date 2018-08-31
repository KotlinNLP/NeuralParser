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
import com.kotlinnlp.syntaxdecoder.utils.removeLast

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
  private val scoresMap: Map<Int, List<ScoredDeprel>>,
  private val maxBeamSize: Int = 10,
  private val maxForkSize: Int = 5,
  private val maxIterations: Int = 10
) {

  /**
   * Raised when there is not a configuration that permits to do not violate a hard constraint.
   */
  class InvalidConfiguration : RuntimeException()

  /**
   * A deprel of a state, including information about its token and its index in the related deprels list.
   *
   * @property tokenId the id of the token of this deprel
   * @property index the index of this deprel within the possible deprels of its token
   * @property deprel the scored deprel
   */
  private data class StateDeprel(val tokenId: Int, val index: Int, var deprel: ScoredDeprel)

  /**
   * The state that represents a configuration of deprels for the [dependencyTree].
   *
   * @property deprels a list of state deprels, one per token and sorted by diff score
   * @property forked whether this state has been already forked
   */
  private inner class State(val deprels: List<StateDeprel>, var forked: Boolean = false) {

    /**
     * The global score of the state.
     */
    val score: Double

    /**
     * Whether this state does not violate any hard constraint.
     */
    var isValid: Boolean = true

    /**
     * Apply the linguistic constraints to this state.
     */
    init {

      this.applyConstraints()

      // Attention: the score must be set after the constraints have been applied.
      this.score = this.deprels.sumByDouble { it.deprel.score }
    }

    /**
     * Fork this states into more, each built replacing the deprel of a token with the following in the [scoresDiffMap].
     *
     * @return the list of new states forked from this
     */
    fun fork(): List<State> {

      val states: MutableList<State> = mutableListOf()

      this.deprels.forEachIndexed { i, deprel ->

        val possibleDeprels: List<ScoredDeprel> = this@DeprelConstraintSolver.scoresMap.getValue(deprel.tokenId)

        if (deprel.index < possibleDeprels.lastIndex) {

          val scoresDiff: List<Double> = this@DeprelConstraintSolver.scoresDiffMap.getValue(deprel.tokenId)
          val newDeprels: List<StateDeprel> = this.deprels
            .replace(i, deprel.copy(index = deprel.index + 1, deprel = possibleDeprels[deprel.index + 1]))
            .sortedBy { if (it.index < possibleDeprels.lastIndex) scoresDiff[it.index] else 1.0 }

          states.add(State(newDeprels))
        }

        if (states.size == this@DeprelConstraintSolver.maxForkSize) return@forEachIndexed
      }

      this.forked = true

      return states
    }

    /**
     * Set the deprels of this state into the [dependencyTree].
     */
    fun applyDeprels() {

      this.deprels.forEach {
        dependencyTree.setDeprel(dependent = it.tokenId, deprel = it.deprel.value)
      }

      dependencyTree.score = this.score
    }

    /**
     * @return the hash code of this state
     */
    override fun hashCode(): Int = this.deprels.map { it.deprel.hashCode() }.hashCode()

    /**
     * @param other any other object
     *
     * @return whether this state is equal to the given object
     */
    override fun equals(other: Any?): Boolean {

      if (this === other) return true
      if (javaClass != other?.javaClass) return false

      other as State

      if (deprels != other.deprels) return false

      return true
    }

    /**
     * Apply the [constraints] to this state.
     */
    private fun applyConstraints() {

      this.applyDeprels()

      val morphoSyntacticTokens: Map<Int, MorphoSyntacticToken> =
        sentence.tokens.associate { it.id to it.toMorphoSyntactic() }

      constraints.forEach { constraint ->
        this.deprels.forEach {

          val dependent: MorphoSyntacticToken = morphoSyntacticTokens.getValue(it.tokenId)
          val governor: MorphoSyntacticToken? =
            dependencyTree.getHead(it.tokenId)?.let { morphoSyntacticTokens.getValue(it) }

          val isVerified: Boolean = when (constraint) {
            is DoubleConstraint ->
              constraint.isVerified(dependent = dependent, governor = governor, dependencyTree = dependencyTree)
            is SingleConstraint -> constraint.isVerified(token = dependent, dependencyTree = dependencyTree)
            else -> throw RuntimeException("Invalid Constraint type: ${constraint.javaClass.simpleName}")
          }

          if (!isVerified) {
            if (constraint.isHard) this.isValid = false
            it.deprel = it.deprel.copy(score = it.deprel.score * constraint.penalty)
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

    /**
     * Replace an element of this list at a given index with another returning a new list.
     *
     * @param index the index of the replacement
     * @param elm the replacement
     *
     * @return a new list with the given element replaced
     */
    private fun <T> List<T>.replace(index: Int, elm: T): List<T> =
      this.mapIndexed { i, it ->  if (i == index) elm else it }
  }

  /**
   * The [scoresMap] with the values remapped to the differences between scores of adjacent deprels.
   */
  private val scoresDiffMap: Map<Int, List<Double>> = this.scoresMap.mapValues { (_, deprels) ->
    deprels.zipWithNext().map { it.first.score - it.second.score }
  }

  /**
   * The list of the best states found, sorted by descending score.
   */
  private val beam: MutableList<State> = mutableListOf()

  /**
   * A set of states already visited.
   */
  private val visitedStates: MutableSet<State> = mutableSetOf()

  /**
   * Whether to prohibit invalid states into the beam.
   * It is false when the algorithm starts, in order to find a first valid state even if it is not present into the
   * first forks.
   * When a valid state is found, then it is set to true.
   */
  private var validStatesOnly = false

  /**
   * Find the best configuration of deprels for the given [dependencyTree] that does not violates any hard constraint,
   * using the given [scoresMap].
   *
   * @throws InvalidConfiguration if all the possible deprels configurations violate a hard constraint
   */
  fun solve() {

    this.initBeam()

    run steps@{
      (0 until this.maxIterations).forEach { if (!this.solvingStep()) return@steps }
    }

    this.beam.firstOrNull()?.applyDeprels() ?: throw InvalidConfiguration()
  }

  /**
   * Initialize the beam with the first state (the one with highest score deprels).
   */
  private fun initBeam() {

    val initState = State(
      deprels = this.scoresMap.map { StateDeprel(tokenId = it.key, index = 0, deprel = it.value.first()) })

    if (initState.isValid) this.validStatesOnly = true

    this.beam.add(initState)
  }

  /**
   * Let the beam go forward to the next step, forking each state into more.
   *
   * @return true if at least one new state has been added to the beam, otherwise false
   */
  private fun solvingStep(): Boolean {

    val forkedStates: List<State> = this.beam.filterNot { it.forked }.flatMap { it.fork() }

    if (!this.validStatesOnly && forkedStates.any { it.isValid }) {

      this.validStatesOnly = true

      this.beam.retainAll { it.isValid } // invalid states are no more allowed -> remove them from the beam
    }

    return this.getAllowedStates(forkedStates).map { this.addNewState(it) }.any { it }
  }

  /**
   * @param states a list of states
   *
   * @return the states that are allowed to enter the beam
   */
  private fun getAllowedStates(states: List<State>): List<State> =
    if (this.validStatesOnly) states.filter { it.isValid } else states

  /**
   * Add a new state (if possible) into the beam.
   *
   * @param state the new state to add
   *
   * @return true if a new state has been added, otherwise false
   */
  private fun addNewState(state: State): Boolean {

    if (state in this.visitedStates) return false

    val beamFull: Boolean = this.beam.size == this.maxBeamSize

    if (!beamFull || state.score > this.beam.last().score) {

      val index: Int = this.beam.indexOfFirst { it.score < state.score }.let { if (it >= 0) it else this.beam.size }

      if (beamFull) this.beam.removeLast()

      this.beam.add(index, state)
      this.visitedStates.add(state)

      return true
    }

    return false
  }
}
