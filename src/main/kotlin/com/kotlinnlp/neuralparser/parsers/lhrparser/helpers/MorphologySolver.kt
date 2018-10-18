/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken

/**
 * A helper that finds all the valid morphologies of morpho-syntactic tokens respect to given constraints.
 *
 * @param tokens a list of single morpho-syntactic tokens
 * @param constraints a list of linguistic constraints
 */
internal class MorphologySolver(
  private val tokens: List<MorphoSynToken.Single>,
  private val constraints: List<Constraint>
) : BeamManager<MorphologySolver.MorphologyValue, MorphologySolver.MorphoState>(
  valuesMap = tokens.associate { token ->
    token.id to token.morphologies.asSequence().map { MorphologyValue(it) }.sortedByDescending { it.score }.toList()
  },
  maxBeamSize = -1,
  maxForkSize = -1,
  maxIterations = -1
) {

  /**
   * A value containing a scored morphology.
   *
   * @property morphology a scored morphology of a token
   * @property isValid whether the grammatical configuration violates a hard constraint or not
   */
  internal data class MorphologyValue(val morphology: ScoredSingleMorphology, var isValid: Boolean = true) : Value() {

    /**
     * The score of this value.
     */
    override var score: Double = this.morphology.score
  }

  /**
   * The state that represents a configuration of the morphologies of the tokens.
   *
   * @param elements the list of elements in this state, sorted by diff score
   */
  internal inner class MorphoState(elements: List<StateElement<MorphologyValue>>) : State(elements) {

    /**
     * The global score of the state.
     */
    override val score: Double

    /**
     * Whether this state does not violate any hard constraint.
     */
    override var isValid: Boolean = true

    /**
     * The elements of this state associated by id.
     */
    private val elementsById: Map<Int, StateElement<MorphologyValue>> = this.elements.associateBy { it.id }

    /**
     * Apply the linguistic constraints to this state and set its score.
     */
    init {

      // Attention: do not modify the order of the operations.
      this.applyMorphologies()
      this.applyConstraints()

      this.score = this.elements.sumByDouble { it.value.score }
    }

    /**
     * Apply the morphologies of this state to the tokens.
     */
    private fun applyMorphologies() {

      tokens.forEach { token ->
        token.removeAllMorphologies()
        token.addMorphology(this.elementsById.getValue(token.id).value.morphology)
      }
    }

    /**
     * Apply the [constraints] to this state.
     */
    private fun applyConstraints() {

      constraints.forEach { constraint ->
        tokens.forEach { token ->

          val isVerified: Boolean =
            constraint.isVerified(token = token, tokens = tokens, dependencyTree = dependencyTree)

          val element: StateElement<MorphologyValue> = this.elementsById.getValue(token.id)

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
  }

  /**
   * The dependency tree that represents the dependencies of the tokens.
   */
  private val dependencyTree = DependencyTree(this.tokens)

  /**
   * Find all the valid configurations of morphologies for the given [tokens] that do not violate any hard constraint.
   *
   * @return all the valid scored single morphologies for each token or null if no valid configuration has been found
   */
  fun solve(): Map<Int, List<ScoredSingleMorphology>> {

    val morphologiesByToken: MutableMap<Int, MutableList<ScoredSingleMorphology>> =
      tokens.associate { it.id to mutableListOf<ScoredSingleMorphology>() }.toMutableMap()

    this.findConfigurations(onlyValid = true).forEach { state ->
      state.elements.forEach { elm ->
        morphologiesByToken.getValue(elm.id).add(elm.value.morphology.copy(score = elm.value.score))
      }
    }

    return morphologiesByToken.mapValues { it.value.toList() }
  }

  /**
   * Build a new state with the given elements.
   *
   * @param elements the elements that compose the building state
   *
   * @return a new state with the given elements
   */
  override fun buildState(elements: List<StateElement<MorphologyValue>>): MorphoState = MorphoState(elements)
}
