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
 * The Constraint that is applied to a single token.
 *
 * @property description a string that describes the constraint
 * @property penalty the penalty that this constraint gives in case it is violated
 * @property boost the boost that this constraint gives in case it is verified
 * @param premise the premise that permits to verify this constraint
 * @param condition the condition that verifies this constraint
 */
class SingleConstraint(
  description: String,
  penalty: Double,
  boost: Double,
  private val premise: Condition,
  private val condition: Condition
) : Constraint(description = description, penalty = penalty, boost = boost) {

  /**
   * Build a [SingleConstraint] from a JSON string object.
   *
   * @param jsonObject the JSON object of the constraint
   */
  constructor(jsonObject: JsonObject): this(
    description = jsonObject.string("description")!!,
    penalty = getPenalty(jsonObject),
    boost = getBoost(jsonObject),
    premise = Condition(jsonObject.obj("premise")!!),
    condition = Condition(jsonObject.obj("condition")!!)
  )

  /**
   * Whether this constraint needs to look at the context morphology of a token.
   */
  override val checkContext: Boolean = this.premise.checkContext || this.condition.checkContext

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
                          dependencyTree: DependencyTree): Boolean =
    !this.premise.isVerified(token = token, tokens = tokens, dependencyTree = dependencyTree) ||
      this.condition.isVerified(token = token, tokens = tokens, dependencyTree = dependencyTree)
}
