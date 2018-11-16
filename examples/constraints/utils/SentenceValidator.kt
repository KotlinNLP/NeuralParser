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
 * @param constraints a list of linguistic constraints
 * @param morphoPercolator a percolator of morphology properties
 */
internal class SentenceValidator(
  tokens: List<MorphoSynToken.Single>,
  private val constraints: List<Constraint>,
  private val morphoPercolator: MorphoPercolator
) : ConstraintsValidator(tokens) {

  /**
   * The input tokens associated to their list index.
   */
  private val indexedTokens: Iterable<IndexedValue<MorphoSynToken.Single>> = tokens.withIndex()

  /**
   * A list of constraints that do not check any morphology.
   */
  private val simpleConstraints: List<Constraint> = this.constraints.filter { !it.checkMorpho }

  /**
   * A list of constraints that check the base morphology of a single token.
   */
  private val morphoUnaryConstraints: List<Constraint> =
    this.constraints.filter { it.checkMorpho && !it.checkContext && it.isUnary }

  /**
   * A list of constraints that check the base morphology.
   */
  private val morphoConstraints: List<Constraint> =
    this.constraints.filter { it.checkMorpho && !it.checkContext && !it.isUnary }

  /**
   * A list of constraints that check the context morphology.
   */
  private val contextConstraints: List<Constraint> = this.constraints.filter { it.checkContext }

  /**
   * All the possible morphologies for each token.
   */
  private val tokensMorphologies: List<List<ScoredSingleMorphology>> = this.tokens.map { it.morphologies }

  /**
   * A validator of the morphological configuration of a sentence.
   */
  private val morphoValidator = MorphoValidator(
    tokens = this.tokens,
    morphoConstraints = this.morphoConstraints,
    contextConstraints = this.contextConstraints,
    morphoPercolator = this.morphoPercolator)

  /**
   * Find all the constraints violated by each token.
   *
   * @return the map of all the linguistic constraints violated by the input sentence
   */
  fun validate(): ViolationsMap {

    val violationsMap: ViolationsMap = this.verifyConstraints(this.simpleConstraints).notEmptyOr {
      this.getUnaryMorphoViolations().notEmptyOr {
        this.getMorphoViolations()
      }
    }

    return violationsMap.mapValues { it.value.asSequence().toSet().toList() }
  }

  /**
   * @return the map of all the unary linguistic constraints violated by the input sentence
   */
  private fun getUnaryMorphoViolations(): ViolationsMap =
    this.tokens.asSequence().collectViolations { this.getUnaryMorphoViolations(it) }

  /**
   * @param token an input token
   *
   * @return the map of all the unary linguistic constraints violated by the given token
   */
  private fun getUnaryMorphoViolations(token: MorphoSynToken.Single): ViolationsMap {

    val morphologies: List<ScoredSingleMorphology> = token.morphologies

    val violations: ViolationsMap = morphologies.asSequence().collectViolations { morphology ->

      token.setMorphology(morphology) // set a single morphology per iteration

      val violations: List<Constraint> = this.morphoUnaryConstraints.filter {
        !it.isVerified(token = token, tokens = this.tokens, dependencyTree = this.dependencyTree)
      }

      if (violations.isNotEmpty()) mapOf(token.id to violations) else mapOf()
    }

    token.setMorphologies(morphologies) // restore all morphologies

    return violations
  }

  /**
   * @return the map of all the morphological linguistic constraints violated by the input sentence
   */
  private fun getMorphoViolations(): ViolationsMap =
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
