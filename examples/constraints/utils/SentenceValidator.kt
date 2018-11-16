/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.linguisticdescription.morphology.POS
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
  private val allPossibleMorphologies: List<List<ScoredSingleMorphology>> = tokens.map { it.morphologies }

  /**
   * The sets of morphologies that are representative of the valid POS types for each token, associated by token id.
   * This sets are reduced during the execution, removing morphologies related to POS types that do not verify all the
   * required constraints.
   */
  private val validPOSMorphoMap: Map<Int, MutableSet<ScoredSingleMorphology>> = this.tokens.associate { token ->

    val posSet: Set<POS> = token.morphologies.map { it.value.pos }.toSet()

    token.id to posSet.map { pos -> token.morphologies.first { it.value.pos == pos } }.toMutableSet()
  }

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
        this.getBaseMorphoBinaryViolations().notEmptyOr {
          this.getBaseMorphoOtherViolations().notEmptyOr {
            this.getMorphoPropertiesViolations()
          }
        }
      }
    }

    return violationsMap.mapValues { it.value.asSequence().toSet().toList() }
  }

  /**
   * @return the map of all the unary linguistic constraints of base morphology violated by the input sentence
   */
  private fun getBaseMorphoUnaryViolations(): ViolationsMap =
    this.tokens.asSequence().collectViolations(clearOnValid = false) { this.getBaseMorphoUnaryViolations(it) }

  /**
   * @param token an input token
   *
   * @return the map of all the unary linguistic constraints of base morphology violated by the given token
   */
  private fun getBaseMorphoUnaryViolations(token: MorphoSynToken.Single): ViolationsMap =

    this.validPOSMorphoMap.getValue(token.id).toList().asSequence().collectViolations { morphology ->

      token.setMorphology(morphology) // set a single morphology per iteration

      val violatedConstraints: List<Constraint> =
        this.verifyConstraints(token = token, constraints = this.groupedConstraint.baseMorphoUnary)

      if (violatedConstraints.isNotEmpty()) {
        this.validPOSMorphoMap.getValue(token.id).remove(morphology)
        mapOf(token.id to violatedConstraints)
      } else {
        mapOf()
      }
    }

  /**
   * @return the map of all the binary linguistic constraints of base morphology violated by the input sentence
   */
  private fun getBaseMorphoBinaryViolations(): ViolationsMap =

    this.tokens.asSequence().collectViolations(clearOnValid = false) {

      val governorId: Int? = this.dependencyTree.getHead(it.id)

      if (governorId != null)
        this.getBaseMorphoBinaryViolations(
          dependent = it,
          governor = this.tokens[this.dependencyTree.getPosition(governorId)])
      else
        mapOf()
    }

  /**
   * @param dependent a dependent token
   * @param governor a governor token
   *
   * @return the map of all the binary linguistic constraints of base morphology violated by the given tokens
   */
  private fun getBaseMorphoBinaryViolations(dependent: MorphoSynToken.Single,
                                            governor: MorphoSynToken.Single): ViolationsMap {

    val depValidMorphologies: MutableSet<ScoredSingleMorphology> = mutableSetOf()
    val govValidMorphologies: MutableSet<ScoredSingleMorphology> = mutableSetOf()

    val violations: ViolationsMap = this.getPOSMorphoCombinations(dependent = dependent, governor = governor)
      .collectViolations { (depMorpho, govMorpho) ->

        dependent.setMorphology(depMorpho) // set a single morphology per iteration
        governor.setMorphology(govMorpho) // set a single morphology per iteration

        val violatedConstraints: List<Constraint> =
          this.verifyConstraints(token = dependent, constraints = this.groupedConstraint.baseMorphoBinary)

        if (violatedConstraints.isNotEmpty()) {
          mapOf(dependent.id to violatedConstraints)
        } else {
          depValidMorphologies.add(depMorpho)
          govValidMorphologies.add(govMorpho)
          mapOf()
        }
      }

    this.validPOSMorphoMap.getValue(dependent.id).retainAll(depValidMorphologies)
    this.validPOSMorphoMap.getValue(governor.id).retainAll(govValidMorphologies)

    return violations
  }

  /**
   * @param dependent a dependent token
   * @param governor a governor token
   *
   * @return the sequence of all the valid POS representative morphologies combinations for the given tokens
   */
  private fun getPOSMorphoCombinations(
    dependent: MorphoSynToken.Single,
    governor: MorphoSynToken.Single
  ): Sequence<Pair<ScoredSingleMorphology, ScoredSingleMorphology>> {

    val depMorphologies: Sequence<ScoredSingleMorphology> = this.validPOSMorphoMap.getValue(dependent.id).asSequence()
    val govMorphologies: Sequence<ScoredSingleMorphology> = this.validPOSMorphoMap.getValue(governor.id).asSequence()

    return depMorphologies.flatMap { depMorpho -> govMorphologies.map { govMorpho -> Pair(depMorpho, govMorpho) } }
  }

  /**
   * @return the map of all the other linguistic constraints of base morphology violated by the input sentence
   */
  private fun getBaseMorphoOtherViolations(): ViolationsMap {

    val possibleMorphologies: List<List<ScoredSingleMorphology>> =
      this.tokens.map { this.validPOSMorphoMap.getValue(it.id).toList() }

    return this.getAllMorphoConfigurations(possibleMorphologies).collectViolations { morphoConfig ->
      this.tokens.applyMorphologies(morphoConfig)
      this.verifyConstraints(this.groupedConstraint.baseMorphoOthers)
    }
  }

  /**
   * @return the map of all the linguistic constraints that check morphological properties violated by the input
   *         sentence
   */
  private fun getMorphoPropertiesViolations(): ViolationsMap {

    val validMorphologies: List<List<ScoredSingleMorphology>> =
      this.tokens.zip(this.allPossibleMorphologies).map { (token, allMorphologies) ->
        val validPosSet: Set<POS> = this.validPOSMorphoMap.getValue(token.id).map { it.value.pos }.toSet()
        allMorphologies.filter { it.value.pos in validPosSet }
      }

    return this.getAllMorphoConfigurations(validMorphologies).collectViolations {
      this.morphoValidator.getViolations(it)
    }
  }

  /**
   * @param possibleMorphologies the possible morphologies that can be assigned to each token
   * @param focusIndex the index of the focus token within the recursion
   * @param morphoIndices the indices of the focus morphologies, one per token
   *
   * @return all the possible morphological configurations of the input [tokens]
   */
  private fun getAllMorphoConfigurations(
    possibleMorphologies: List<List<ScoredSingleMorphology>>,
    focusIndex: Int = 0,
    morphoIndices: MutableList<Int> = MutableList(size = this.tokens.size, init = { 0 })
  ): Sequence<MorphoConfiguration> = sequence {

    possibleMorphologies[focusIndex].indices.forEach { focusMorphoIndex ->

      morphoIndices[focusIndex] = focusMorphoIndex

      yield(buildMorphoConfiguration(possibleMorphologies = possibleMorphologies, morphoIndices = morphoIndices))

      if (focusIndex < tokens.lastIndex)
        yieldAll(
          getAllMorphoConfigurations(
            possibleMorphologies = possibleMorphologies,
            focusIndex = focusIndex + 1,
            morphoIndices = morphoIndices))
    }
  }

  /**
   * @param possibleMorphologies the possible morphologies that can be assigned to each token
   * @param morphoIndices the indices of the focus morphologies, one per token
   *
   * @return a morphological configuration of the input sentence built using the token morphologies at the given indices
   */
  private fun buildMorphoConfiguration(possibleMorphologies: List<List<ScoredSingleMorphology>>,
                                       morphoIndices: List<Int>): MorphoConfiguration =

    this.indexedTokens.associate { (tokenIndex, token) ->

      val tokenMorphologies: List<ScoredSingleMorphology> = possibleMorphologies[tokenIndex]
      val tokenMorphoIndex: Int = morphoIndices[tokenIndex]

      token.id to tokenMorphologies[tokenMorphoIndex]
    }
}
