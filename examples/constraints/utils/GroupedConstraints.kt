/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.constraints.DoubleConstraint

/**
 * Constraints grouped by different types.
 *
 * @param constraints all the constraints that must be grouped
 */
internal class GroupedConstraints(constraints: List<Constraint>) {

  /**
   * A set of constraints that do not check any morphology.
   */
  val simple: Set<Constraint> = constraints.asSequence().filter { !it.checkMorpho }.toSet()

  /**
   * A set of constraints that check the base morphology of a single token.
   */
  val baseMorphoUnary: Set<Constraint> =
    (constraints - simple)
      .asSequence()
      .filter { !it.checkMorphoProp && it.isUnary }
      .toSet()

  /**
   * A set of constraints that check the base morphology of a dependent-governor tokens pair.
   */
  val baseMorphoBinary: Set<DoubleConstraint> =
    (constraints - simple - baseMorphoUnary)
      .asSequence()
      .filter { !it.checkMorphoProp && it.isBinary }
      .map { it as DoubleConstraint }
      .toSet()

  /**
   * A set of constraints that check the base morphology of tokens at major distance.
   */
  val baseMorphoOthers: Set<Constraint> =
    (constraints - simple - baseMorphoUnary - baseMorphoBinary)
      .asSequence()
      .filter { !it.checkMorphoProp }
      .toSet()

  /**
   * A set of constraints that check the morphological properties but not the context ones.
   */
  val morphoPropSimple: Set<Constraint> =
    (constraints - simple - baseMorphoUnary - baseMorphoBinary - baseMorphoOthers)
      .asSequence()
      .filter { it.checkMorphoProp && !it.checkContext }
      .toSet()

  /**
   * A set of constraints that check the context morphological properties.
   */
  val morphoPropContext: Set<Constraint> =
    (constraints - simple - baseMorphoUnary - baseMorphoBinary - baseMorphoOthers - morphoPropSimple)
    .asSequence()
    .filter { it.checkContext }
    .toSet()

  /**
   * Check requirements.
   */
  init {

    val allGroups: Sequence<Set<Constraint>> =
      sequenceOf(simple, baseMorphoUnary, baseMorphoBinary, baseMorphoOthers, morphoPropSimple, morphoPropContext)

    require(allGroups.sumBy { it.size } == constraints.size) { "The constraints groups must be complementary." }
  }
}
