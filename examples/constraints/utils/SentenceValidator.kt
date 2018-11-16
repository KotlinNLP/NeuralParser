/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.morphopercolator.MorphoPercolator
import com.kotlinnlp.utils.notEmptyOr

/**
 * A helper that validate constraints on all the morphologies combinations of a list of tokens.
 *
 * @param tokens a list of single morpho-syntactic tokens
 * @param groupedConstraint linguistic constraints grouped by different types
 * @param morphoPercolator a percolator of morphology properties
 */
internal class SentenceValidator(
  tokens: List<MorphoSynToken.Single>,
  private val groupedConstraint: GroupedConstraints,
  private val morphoPercolator: MorphoPercolator
) : ConstraintsValidator(tokens) {

  /**
   * The input tokens associated to their list index.
   */
  private val indexedTokens: Iterable<IndexedValue<MorphoSynToken.Single>> = tokens.withIndex()

  /**
   * All the possible morphologies for each token.
   */
  private val tokensMorphologies: List<List<ScoredSingleMorphology>> = this.tokens.map { it.morphologies }

  /**
   * A validator of the morphological configuration of a sentence.
   */
  private val morphoValidator = MorphoValidator(
    tokens = this.tokens,
    groupedConstraint = this.groupedConstraint,
    morphoPercolator = this.morphoPercolator)

  /**
   * Find all the constraints violated by each token.
   *
   * @return the map of all the linguistic constraints violated by the input sentence
   */
  fun validate(): ViolationsMap {

    val violationsMap: ViolationsMap = this.verifyConstraints(this.groupedConstraint.simple).notEmptyOr {
      this.getBaseMorphoUnaryViolations().notEmptyOr {
        this.getMorphoPropertiesViolations()
      }
    }

    return violationsMap.mapValues { it.value.asSequence().toSet().toList() }
  }

  /**
   * @return the map of all the unary linguistic constraints violated by the input sentence
   */
  private fun getBaseMorphoUnaryViolations(): ViolationsMap =
    this.tokens.asSequence().collectViolations { this.getBaseMorphoUnaryViolations(it) }

  /**
   * @param token an input token
   *
   * @return the map of all the unary linguistic constraints violated by the given token
   */
  private fun getBaseMorphoUnaryViolations(token: MorphoSynToken.Single): ViolationsMap {

    val morphologies: Sequence<ScoredSingleMorphology> = token.morphologies.asSequence()

    val violations: ViolationsMap = morphologies.collectViolations { morphology ->

      token.setMorphology(morphology) // set a single morphology per iteration

      val violations: List<Constraint> = this.groupedConstraint.baseMorphoUnary.filter {
        !it.isVerified(token = token, tokens = this.tokens, dependencyTree = this.dependencyTree)
      }

      if (violations.isNotEmpty()) mapOf(token.id to violations) else mapOf()
    }

    token.setMorphologies(morphologies) // restore all morphologies

    return violations
  }

  /**
   * @return the map of all the linguistic constraints that check morphological properties violated by the input
   *         sentence
   */
  private fun getMorphoPropertiesViolations(): ViolationsMap =
    this.getAllMorphoConfigurations().collectViolations { this.morphoValidator.getViolations(it) }

  /**
   * @param focusIndex the index of the focus token within the recursion
   * @param morphoIndices the indices of the focus morphologies, one per token
   *
   * @return all the possible morphological configurations of the input [tokens]
   */
  private fun getAllMorphoConfigurations(
    focusIndex: Int = 0,
    morphoIndices: MutableList<Int> = MutableList(size = this.tokens.size, init = { 0 })
  ): Sequence<MorphoConfiguration> = sequence {

    tokensMorphologies[focusIndex].indices.forEach { focusMorphoIndex ->

      morphoIndices[focusIndex] = focusMorphoIndex

      yield(buildMorphoConfiguration(morphoIndices))

      if (focusIndex < tokens.lastIndex)
        yieldAll(getAllMorphoConfigurations(focusIndex = focusIndex + 1, morphoIndices = morphoIndices))
    }
  }

  /**
   * @param morphoIndices the indices of the focus morphologies, one per token
   *
   * @return a morphological configuration of the input sentence built using the token morphologies at the given indices
   */
  private fun buildMorphoConfiguration(morphoIndices: List<Int>): MorphoConfiguration =

    this.indexedTokens.associate { (tokenIndex, token) ->

      val tokenMorphologies: List<ScoredSingleMorphology> = this.tokensMorphologies[tokenIndex]
      val tokenMorphoIndex: Int = morphoIndices[tokenIndex]

      token.id to tokenMorphologies[tokenMorphoIndex]
    }
}
