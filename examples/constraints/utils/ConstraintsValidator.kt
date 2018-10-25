/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.utils.BeamManager

/**
 * A helper that validate constraints on all the morphologies combinations of a list of tokens.
 *
 * @param tokens a list of single morpho-syntactic tokens
 * @param constraints a list of linguistic constraints
 */
internal class ConstraintsValidator(
  private val tokens: List<MorphoSynToken.Single>,
  private val constraints: List<Constraint>
) : BeamManager<ConstraintsValidator.MorphologyValue, ConstraintsValidator.MorphoState>(
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
   */
  internal data class MorphologyValue(val morphology: ScoredSingleMorphology) : Value() {

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
    override val score: Double = 1.0

    /**
     * Whether this state does not violate any hard constraint.
     */
    override var isValid: Boolean = true

    /**
     * The map of token ids to the list of constraints that violate in this state.
     */
    val violatedConstraints: MutableMap<Int, MutableList<Constraint>> = mutableMapOf()

    /**
     * Apply the linguistic constraints to this state and set its score.
     */
    init {

      // Attention: do not modify the order of the operations.
      this.applyMorphologies()
      this.applyConstraints()
    }

    /**
     * Apply the morphologies of this state to the tokens.
     */
    private fun applyMorphologies() {

      tokens.zip(this.elements).forEach { (token, element) ->
        token.removeAllMorphologies()
        token.addMorphology(element.value.morphology)
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

          if (!isVerified && constraint.isHard) {
            this.violatedConstraints.getOrPut(token.id) { mutableListOf() }.add(constraint)
            this.isValid = false
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
   * Find all the constraints violated by each token.
   *
   * @return a map of tokens to the list of constraints that they violate
   */
  fun validate(): Map<MorphoSynToken.Single, List<Constraint>> =

    this.findConfigurations(onlyValid = true).fold(mutableMapOf()) { retMap, state ->

      state.violatedConstraints.forEach { tokenId, constraint ->

        val token: MorphoSynToken.Single = this.tokens[this.dependencyTree.getPosition(tokenId)]

        retMap.merge(token, constraint) { t, u -> t + u }
      }

      retMap
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
