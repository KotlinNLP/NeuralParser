/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.constraints

import com.beust.klaxon.JsonObject
import com.kotlinnlp.linguisticconditions.Condition
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken

/**
 * The Constraint that is applied to a pair of dependent and governor.
 *
 * @property description a string that describes the constraint
 * @property penalty the penalty that this constraint gives in case it is violated
 * @property boost the boost that this constraint gives in case it is verified
 * @param premise the premise that permits to verify this constraint
 * @param condition the condition that verifies this constraint
 */
class DoubleConstraint(
  description: String,
  penalty: Double,
  boost: Double,
  private val premise: DoubleCondition,
  private val condition: DoubleCondition
) : Constraint(description = description, penalty = penalty, boost = boost) {

  /**
   * A double condition on a dependent and a governor (at least one must be not null).
   *
   * @property dependent the condition on the dependent (optional)
   * @property governor the condition on the governor (optional)
   */
  data class DoubleCondition(val dependent: Condition?, val governor: Condition?) {

    /**
     * Check requirements.
     */
    init {
      require(this.dependent != null || this.governor != null) {
        "In a DoubleCondition at least one condition between the dependent and the governor must be not null."
      }
    }

    /**
     * Whether this constraint needs to look at the morphological properties of a token.
     */
    val checkMorpho: Boolean = this.dependent?.checkMorpho ?: false || this.governor?.checkMorpho ?: false

    /**
     * Whether this constraint needs to look at the context morphology of a token.
     */
    val checkContext: Boolean = this.dependent?.checkContext ?: false || this.governor?.checkContext ?: false
  }

  /**
   * Whether this constraint looks at a single token, without requiring to check other tokens properties.
   */
  override val isUnary: Boolean = false

  /**
   * Whether this constraint needs to look at the morphological properties of a token.
   */
  override val checkMorpho: Boolean = this.premise.checkMorpho || this.condition.checkMorpho

  /**
   * Whether this constraint needs to look at the context morphology of a token.
   */
  override val checkContext: Boolean = this.premise.checkContext || this.condition.checkContext

  /**
   * Build a [DoubleConstraint] from a JSON string object.
   *
   * @param jsonObject the JSON object of the constraint
   */
  constructor(jsonObject: JsonObject): this(
    description = jsonObject.string("description")!!,
    penalty = getPenalty(jsonObject),
    boost = getBoost(jsonObject),
    premise = DoubleCondition(
      dependent = jsonObject.obj("premise")!!.obj("dependent")?.let { Condition(it) },
      governor = jsonObject.obj("premise")!!.obj("governor")?.let { Condition(it) }
    ),
    condition = DoubleCondition(
      dependent = jsonObject.obj("condition")!!.obj("dependent")?.let { Condition(it) },
      governor = jsonObject.obj("condition")!!.obj("governor")?.let { Condition(it) }
    )
  )

  /**
   * Check if this constraint is verified for a given [token].
   *
   * @param token a token
   * @param tokens the list of all the tokens that compose the sentence
   * @param dependencyTree the dependency tree of the token sentence
   *
   * @return a boolean indicating if this constraint is verified for the given [token]
   */
  override fun isVerified(token: MorphoSynToken.Single,
                          tokens: List<MorphoSynToken.Single>,
                          dependencyTree: DependencyTree): Boolean {

    val governor: MorphoSynToken.Single? = dependencyTree.getHead(token.id)?.let {
      tokens[dependencyTree.getPosition(it)]
    }

    return !this.isPremiseVerified(token, governor, tokens, dependencyTree) ||
      this.isConditionVerified(token, governor, tokens, dependencyTree)
  }

  /**
   * Check if the premise of this constraint is verified for a pair of [dependent] and [governor].
   *
   * @param dependent a dependent token
   * @param governor a governor token or null if called on the root
   * @param tokens the list of all the tokens that compose the sentence
   * @param dependencyTree the dependency tree of the token sentence
   *
   * @return a boolean indicating if the premise of this constraint is verified for the given pair of tokens
   */
  private fun isPremiseVerified(dependent: MorphoSynToken.Single,
                                governor: MorphoSynToken.Single?,
                                tokens: List<MorphoSynToken.Single>,
                                dependencyTree: DependencyTree): Boolean =
    this.premise.dependent?.isVerified(token = dependent, tokens = tokens, dependencyTree = dependencyTree) ?: true &&
      this.premise.governor?.isVerified(token = governor, tokens = tokens, dependencyTree = dependencyTree) ?: true

  /**
   * Check if the condition of this constraint is verified for a pair of [dependent] and [governor].
   *
   * @param dependent a dependent token
   * @param governor a governor token or null if called on the root
   * @param tokens the list of all the tokens that compose the sentence
   * @param dependencyTree the dependency tree of the token sentence
   *
   * @return a boolean indicating if the condition of this constraint is verified for the given pair of tokens
   */
  private fun isConditionVerified(dependent: MorphoSynToken.Single,
                                  governor: MorphoSynToken.Single?,
                                  tokens: List<MorphoSynToken.Single>,
                                  dependencyTree: DependencyTree): Boolean =
    this.condition.dependent?.isVerified(token = dependent, tokens = tokens, dependencyTree = dependencyTree) ?: true &&
      this.condition.governor?.isVerified(token = governor, tokens = tokens, dependencyTree = dependencyTree) ?: true
}
