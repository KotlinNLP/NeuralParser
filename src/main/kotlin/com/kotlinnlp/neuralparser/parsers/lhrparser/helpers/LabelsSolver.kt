/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.constraints.Constraint
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Unknown
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.selector.LabelerSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar

/**
 * A helper that finds the best configuration of the labels of a dependency tree given the probability distribution of
 * the grammatical configurations for each token.
 *
 * @param sentence a parsing sentence
 * @param dependencyTree the dependency tree of the given sentence
 * @param constraints a list of linguistic constraints
 * @param labelerSelector a labeler selector used to build a [MorphoSynToken]
 * @param scoresMap a map of valid deprels (sorted by descending score) associated to each token id
 * @param maxBeamSize the max number of parallel states that the beam supports (-1 = infinite)
 * @param maxForkSize the max number of forks that can be generated from a state (-1 = infinite)
 * @param maxIterations the max number of iterations of solving steps (it is the depth of beam recursion, -1 = infinite)
 */
internal class LabelsSolver(
  private val sentence: ParsingSentence,
  private val dependencyTree: DependencyTree,
  private val constraints: List<Constraint>,
  private val labelerSelector: LabelerSelector,
  scoresMap: Map<Int, List<ScoredGrammar>>,
  maxBeamSize: Int = 5,
  maxForkSize: Int = 3,
  maxIterations: Int = 10
) : BeamManager<LabelsSolver.GrammarValue, LabelsSolver.GrammarState>(
  valuesMap = scoresMap.mapValues { (_, grammar) ->
    grammar.asSequence().map { GrammarValue(it) }.sortedByDescending { it.score }.toList()
  },
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
   * @property isValid whether the grammatical configuration violates a hard constraint or not
   */
  internal data class GrammarValue(val grammar: ScoredGrammar, var isValid: Boolean = true) : Value() {

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
     * Apply the linguistic constraints to this state and apply its configuration to the dependency tree.
     */
    init {

      // Attention: the configuration must be applied before the constraints
      applyConfiguration(this)

      this.applyConstraints()

      // Attention: the score must be set after the constraints have been applied.
      this.score = this.elements.sumByDouble { it.value.score * dependencyTree.getAttachmentScore(it.id) }
    }

    /**
     * Apply the [constraints] to this state.
     */
    private fun applyConstraints() {

      val morphoSynSentence: MorphoSynSentence = sentence.toMorphoSynSentence(
        dependencyTree = this@LabelsSolver.dependencyTree,
        labelerSelector = labelerSelector)

      val explodedTokensPairs: List<Pair<Int, MorphoSynToken>> = this.explodeTokens(morphoSynSentence.tokens)
      val explodedTokens: List<MorphoSynToken> = explodedTokensPairs.map { it.second }
      val dependencyTree = DependencyTree(explodedTokens)

      val elementsById: Map<Int, StateElement<GrammarValue>> = this.elements.associateBy { it.id }

      constraints.forEach { constraint ->
        explodedTokensPairs.forEach { (originalId, token) ->

          val isVerified: Boolean = constraint.isVerified(
            token = token,
            tokens = explodedTokens,
            dependencyTree = dependencyTree)

          val element: StateElement<GrammarValue> = elementsById.getValue(originalId)

          if (!isVerified) {

            if (constraint.isHard) {
              element.value.isValid = false
              this.isValid = false
            }

            element.value.score *= element.value.score * constraint.penalty
          }
        }
      }
    }

    /**
     * Explode a list of morpho-syntactic tokens into a list of Single tokens, adding all the components of the
     * Composite ones.
     *
     * @param tokens a list of morpho-syntactic tokens
     *
     * @return a new list with all the Single tokens of the given list, in pairs with the original token id
     */
    private fun explodeTokens(tokens: List<MorphoSynToken>): List<Pair<Int, MorphoSynToken.Single>> {

      val pairs: MutableList<Pair<Int, MorphoSynToken.Single>> = mutableListOf()
      var index = 0
      val dependencies: Map<Int, Int?> = this.getDependenciesByComponents(tokens)

      tokens.forEach { token ->
        when (token) {
          is MorphoSynToken.Single -> pairs.add(Pair(token.id, token.copyPositionIndex(index)))
          is MorphoSynToken.Composite -> token.components.forEach { c ->
            pairs.add(Pair(token.id, c.copyPositionIndex(index++)))
          }
        }
      }

      pairs.forEach { (_, token) ->
        token.updateSyntacticRelation(token.syntacticRelation.copy(governor = dependencies.getValue(token.id)))
      }

      return pairs
    }

    /**
     * Get the dependencies of the given tokens considering the components in case of Composite tokens.
     *
     * @param tokens a list of morpho-syntactic tokens
     *
     * @return the map of Single token ids to their heads
     */
    private fun getDependenciesByComponents(tokens: List<MorphoSynToken>): Map<Int, Int?> {

      val dependencies: MutableMap<Int, Int?> = tokens
        .flatMap {
          when (it) {
            is MorphoSynToken.Single -> listOf(it)
            is MorphoSynToken.Composite -> it.components
          }
        }
        .associate { it.id to it.syntacticRelation.governor }
        .toMutableMap()

      tokens.forEach {
        if (it is MorphoSynToken.Composite)
          dependencies
            .filterValues { governorId -> governorId == it.id }
            .keys
            .forEach { dependentId -> dependencies[dependentId] = it.components.first().id }
      }

      return dependencies
    }

    /**
     * Copy a Single Token assigning a new position index to it.
     *
     * @param index the new position index
     *
     * @return the same token with the position index replaced
     */
    private fun MorphoSynToken.Single.copyPositionIndex(index: Int): MorphoSynToken.Single = when (this) {
      is Word -> Word(
        id = this.id,
        form = this.form,
        position = this.position.copy(index = index),
        pos = this.pos,
        morphologies = this.morphologies,
        contextMorphologies = this.contextMorphologies,
        syntacticRelation = this.syntacticRelation,
        coReferences = this.coReferences,
        semanticRelations = this.semanticRelations)
      is WordTrace -> this
      else -> throw RuntimeException("Invalid token type: only Word and WordTrace are allowed at this step.")
    }
  }

  /**
   * Find the best grammatical configuration for the given [dependencyTree] that does not violates any hard constraint,
   * using the given scores map.
   *
   * @throws InvalidConfiguration if all the possible configurations violate a hard constraint
   */
  fun solve() {

    val bestConfig: GrammarState = this.findBestConfiguration(onlyValid = false)!!

    this.applyConfiguration(bestConfig)

    this.dependencyTree.score = bestConfig.score
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
   * Set the grammatical configuration of a given state into the [dependencyTree].
   *
   * @param state a state
   */
  private fun applyConfiguration(state: GrammarState) {

    state.elements.forEach {
      this.dependencyTree.setGrammaticalConfiguration(
        dependent = it.id,
        configuration = if (it.value.isValid) it.value.grammar.config else it.value.grammar.config.toUnknown())
    }
  }

  /**
   * @return a new grammatical configuration with unknown values built from this
   */
  private fun GrammaticalConfiguration.toUnknown() = GrammaticalConfiguration(this.components.map {
    GrammaticalConfiguration.Component(
      syntacticDependency = Unknown(direction = it.syntacticDependency.direction),
      pos = null)
  })
}
