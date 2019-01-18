/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.constraints.Constraint

/**
 * The validator of a the base morphology of a pair of dependent-governor tokens.
 *
 * @param tokens the list of single morpho-syntactic tokens that compose a sentence
 * @param dependent the dependent token
 * @param governor the governor token
 * @param groupedConstraint linguistic constraints grouped by different types
 * @param validPOSMorphoMap
 */
internal class BaseMorphoBinaryValidator(
  tokens: List<MorphoSynToken.Single>,
  private val dependent: MorphoSynToken.Single,
  private val governor: MorphoSynToken.Single,
  private val groupedConstraint: GroupedConstraints,
  private val validPOSMorphoMap: Map<Int, MutableSet<ScoredSingleMorphology>>
) : ConstraintsValidator(tokens) {

  /**
   * All the combinations of morphologies of the [dependent] and the [governor].
   */
  private val combinations: Sequence<MorphoPair> = this.buildPOSMorphoCombinations()

  /**
   * The set of visited morphologies combinations.
   */
  private val visitedCombinations: MutableSet<Pair<ScoredSingleMorphology, ScoredSingleMorphology>> = mutableSetOf()

  /**
   * The set of valid morphologies of the [dependent].
   */
  private val depValidMorphologies: MutableSet<ScoredSingleMorphology> = mutableSetOf()

  /**
   * The set of valid morphologies of the [governor].
   */
  private val govValidMorphologies: MutableSet<ScoredSingleMorphology> = mutableSetOf()

  /**
   * @return the map of the morphological linguistic constraints violated by the [dependent]-[governor] pair
   */
  fun getViolations(): ViolationsMap {

    val violations: ViolationsMap = this.combinations.collectViolations { this.getMorphoPairViolations(it) }

    if (violations.isEmpty()) this.completeConstraintsVerification()

    this.updateValidPOSMorphoMap()

    return violations
  }

  /**
   * @return all the combinations of valid POS representative morphologies for the [dependent]-[governor] pair
   */
  private fun buildPOSMorphoCombinations(): Sequence<MorphoPair> {

    val depMorphologies: Sequence<ScoredSingleMorphology> =
      this.validPOSMorphoMap.getValue(this.dependent.id).asSequence()

    val govMorphologies: Sequence<ScoredSingleMorphology> =
      this.validPOSMorphoMap.getValue(this.governor.id).asSequence()

    return depMorphologies.flatMap { depMorpho -> govMorphologies.map { govMorpho -> Pair(depMorpho, govMorpho) } }
  }

  /**
   *
   */
  private fun getMorphoPairViolations(morphoPair: MorphoPair): ViolationsMap {

    this.visitedCombinations.add(morphoPair)

    this.applyCombination(morphoPair)

    val violatedConstraints: List<Constraint> =
      this.verifyConstraints(token = this.dependent, constraints = this.groupedConstraint.baseMorphoBinary)

    return if (violatedConstraints.isNotEmpty()) {
      mapOf(dependent.id to violatedConstraints)
    } else {
      addValidCombination(morphoPair)
      mapOf()
    }
  }

  /**
   *
   */
  private fun completeConstraintsVerification() {

    (this.combinations - this.visitedCombinations).forEach {

      this.applyCombination(it)

      if (this.verifyConstraints().isEmpty())
        addValidCombination(it)
    }
  }

  /**
   *
   */
  private fun updateValidPOSMorphoMap() {

    this.validPOSMorphoMap.getValue(this.dependent.id).retainAll(this.depValidMorphologies)
    this.validPOSMorphoMap.getValue(this.governor.id).retainAll(this.govValidMorphologies)
  }

  /**
   *
   */
  private fun applyCombination(morphoPair: MorphoPair) {

    this.dependent.setMorphology(morphoPair.first)
    this.governor.setMorphology(morphoPair.second)
  }

  /**
   *
   */
  private fun addValidCombination(morphoPair: MorphoPair) {

    this.depValidMorphologies.add(morphoPair.first)
    this.govValidMorphologies.add(morphoPair.second)
  }

  /**
   *
   */
  private fun verifyConstraints(): List<Constraint> =
    this.verifyConstraints(token = dependent, constraints = this.groupedConstraint.baseMorphoBinary)

}
