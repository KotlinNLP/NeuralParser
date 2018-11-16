/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.constraints

import com.beust.klaxon.JsonObject
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken

/**
 * The Constraint.
 *
 * @property description a string that describes the constraint
 * @property penalty the penalty that this constraint gives in case it is violated
 * @property boost the boost that this constraint gives in case it is verified
 */
abstract class Constraint(
  val description: String,
  val penalty: Double,
  val boost: Double
) {

  /**
   * Thrown when a JSON constraint object is not valid.
   *
   * @param message an error message
   */
  class InvalidJSONConstraint(message: String) : RuntimeException(message)

  /**
   * Factory.
   */
  companion object Factory {

    /**
     * Build a [Constraint] from a JSON object.
     *
     * @param jsonObject the JSON object representing a constraint
     *
     * @return a new constraint
     */
    operator fun invoke(jsonObject: JsonObject): Constraint {

      validateJSONConstraint(jsonObject)

      return if (isDoubleConstraint(jsonObject)) DoubleConstraint(jsonObject) else SingleConstraint(jsonObject)
    }

    /**
     * @param jsonObject the JSON object representing a constraint
     *
     * @return the penalty of the constraint
     */
    fun getPenalty(jsonObject: JsonObject): Double = jsonObject.double("penalty") ?: 1.0

    /**
     * @param jsonObject the JSON object representing a constraint
     *
     * @return the boost of the constraint
     */
    fun getBoost(jsonObject: JsonObject): Double = jsonObject.double("boost") ?: 1.0

    /**
     * Validate a JSON constraint.
     *
     * @param jsonObject the JSON object representing a constraint
     *
     * @throws InvalidJSONConstraint when the [jsonObject] does not contain the expected fields
     */
    private fun validateJSONConstraint(jsonObject: JsonObject) {

      if ("penalty" !in jsonObject && "boost" !in jsonObject)
        throw InvalidJSONConstraint("A JSON constraint must contain at least one field between 'penalty' and 'boost'.")

      listOf("description", "premise", "condition").forEach { field ->
        if (field !in jsonObject)
          throw InvalidJSONConstraint("A JSON constraint must contain the '$field' field.")
      }

      if (isDoubleConstraint(jsonObject)) {
        validateDoubleConstraint(jsonObject)
      }
    }

    /**
     * Validate a double constraint (containing the 'governor' or the 'dependent' operators in its 'promise' or
     * 'condition').
     *
     * @param jsonObject the JSON object representing a double constraint
     *
     * @throws InvalidJSONConstraint when the [jsonObject] does not contain the expected fields
     */
    private fun validateDoubleConstraint(jsonObject: JsonObject) {

      listOf("premise", "condition").forEach {
        val conditionObj: JsonObject = jsonObject.obj(it)!!
        if (conditionObj.keys.union(setOf("dependent", "governor")).size != 2)
          throw InvalidJSONConstraint(
            "The '$it' of a double constraint cannot contain other fields in addition to 'dependent' and 'governor'.")
      }
    }

    /**
     * @param jsonObject the JSON object representing a double constraint
     *
     * @return a boolean indicating if the constraint is double
     */
    private fun isDoubleConstraint(jsonObject: JsonObject): Boolean = listOf("premise", "condition").any {
      val conditionObj: JsonObject = jsonObject.obj(it)!!
      "dependent" in conditionObj || "governor" in conditionObj
    }
  }

  /**
   * Whether this constraint is hard (penalty == 0.0).
   */
  val isHard: Boolean = this.penalty == 0.0

  /**
   * Whether this constraint looks at a single token, without requiring to check other tokens properties.
   */
  abstract val isUnary: Boolean

  /**
   * Whether this constraint needs to look at the morphology of a token.
   */
  abstract val checkMorpho: Boolean

  /**
   * Whether this constraint needs to look at the morphological properties of a token.
   */
  abstract val checkMorphoProp: Boolean

  /**
   * Whether this constraint needs to look at the context morphology of a token.
   */
  abstract val checkContext: Boolean

  /**
   * Check requirements.
   */
  init {
    require(this.penalty in 0.0 .. 1.0)
    require(this.boost >= 1.0)
    require(this.penalty != 1.0 || this.boost != 1.0)
  }

  /**
   * @return a string representation of this constraint
   */
  override fun toString(): String = this.description

  /**
   * Check if this constraint is verified for a given [token].
   *
   * @param token a token
   * @param tokens the list of all the tokens that compose the sentence
   * @param dependencyTree the dependency tree of the token sentence
   *
   * @return a boolean indicating if this constraint is verified for the given [token]
   */
  abstract fun isVerified(token: MorphoSynToken.Single,
                          tokens: List<MorphoSynToken.Single>,
                          dependencyTree: DependencyTree): Boolean

  /**
   * @param other another object
   *
   * @return true if this constraint is equal to the other object, otherwise false
   */
  override fun equals(other: Any?): Boolean = other is Constraint && this.description == other.description

  /**
   * @return the hash code of this constraint
   */
  override fun hashCode(): Int = this.description.hashCode()
}
