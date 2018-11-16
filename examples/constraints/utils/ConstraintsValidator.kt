/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.constraints.Constraint

/**
 * Validate constraints on a sentence.
 *
 * @param tokens the list of single morpho-syntactic tokens that compose a sentence
 */
internal abstract class ConstraintsValidator(protected val tokens: List<MorphoSynToken.Single>) {

  /**
   * The dependency tree that represents the dependencies of the input [tokens].
   */
  protected val dependencyTree = DependencyTree(this.tokens)

  /**
   * Apply the given [constraints] to the input sentence.
   *
   * @param constraints a set of linguistic constraints to verify
   *
   * @return the map of the linguistic constraints violated by the input sentence, within the given ones
   */
  protected fun verifyConstraints(constraints: Set<Constraint>): ViolationsMap {

    val violatedConstraints: MutableMap<Int, MutableList<Constraint>> = mutableMapOf()

    constraints.forEach { constraint ->
      this.tokens.forEach { token ->
        if (!constraint.isVerified(token = token, tokens = this.tokens, dependencyTree = this.dependencyTree))
          violatedConstraints.getOrPut(token.id) { mutableListOf() }.add(constraint)
      }
    }

    return violatedConstraints
  }
}
