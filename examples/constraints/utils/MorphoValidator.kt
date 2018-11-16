/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.morphopercolator.MorphoPercolator
import com.kotlinnlp.utils.notEmptyOr

/**
 * The validator of a morphological configuration of a sentence.
 *
 * @param tokens the list of single morpho-syntactic tokens that compose a sentence
 * @param morphoConstraints a set of constraints that check the base morphological properties
 * @param contextConstraints a set of constraints that check the context morphology
 * @param morphoPercolator a morphological properties percolator
 */
internal class MorphoValidator(
  tokens: List<MorphoSynToken.Single>,
  private val morphoConstraints: Set<Constraint>,
  private val contextConstraints: Set<Constraint>,
  private val morphoPercolator: MorphoPercolator
) : ConstraintsValidator(tokens) {

  /**
   * @param morphoConfig a morphological configuration of the input [tokens]
   *
   * @return the map of the morphological linguistic constraints violated by the input tokens, applying the given
   *         morphological configuration
   */
  fun getViolations(morphoConfig: MorphoConfiguration): ViolationsMap {

    this.tokens.applyMorphologies(morphoConfig)

    return this.verifyConstraints(this.morphoConstraints).notEmptyOr { this.getContextViolations() }
  }

  /**
   * @return the map of the morphological context constraints violated by the input tokens
   */
  private fun getContextViolations(): ViolationsMap {

    val contextConfigurations: List<MorphoConfiguration> =
      this.morphoPercolator.applyPercolation(tokens = this.tokens, dependencyTree = this.dependencyTree)

    return contextConfigurations.asSequence().collectViolations { contextConfig ->

      this.applyContextMorphologies(contextConfig)

      this.verifyConstraints(this.contextConstraints)
    }
  }

  /**
   * Apply the given context morphologies to the input [tokens].
   *
   * @param contextMorphoConfig a map of context morphologies associated by token id
   */
  private fun applyContextMorphologies(contextMorphoConfig: MorphoConfiguration) {

    this.tokens.forEach { token ->
      token.removeAllContextMorphologies()
      token.addContextMorphology(contextMorphoConfig.getValue(token.id))
    }
  }
}
