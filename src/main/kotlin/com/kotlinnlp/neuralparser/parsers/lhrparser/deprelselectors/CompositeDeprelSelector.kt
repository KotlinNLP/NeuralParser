/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar
import com.kotlinnlp.neuralparser.utils.notEmptyOr

/**
 * The deprel selector to use with the "composite" deprel format.
 */
class CompositeDeprelSelector : MorphoDeprelSelector {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * The label of the unknown deprel.
     */
    private const val UNKNOWN_LABEL = "NOUN-UNKNOWN"

    /**
     * Get the direction of a syntactic dependency.
     *
     * @param tokenIndex the index of the token to which the deprel must be assigned
     * @param headIndex the index of the token head (can be null)
     *
     * @return the direction of the syntactic dependency between the given token and its head
     */
    private fun getDependencyDirection(tokenIndex: Int, headIndex: Int?): SyntacticDependency.Direction = when {
      headIndex == null -> SyntacticDependency.Direction.ROOT
      tokenIndex < headIndex -> SyntacticDependency.Direction.LEFT
      else -> SyntacticDependency.Direction.RIGHT
    }
  }

  /**
   * Get the list of scored grammatical configurations that are valid for a given attachment.
   *
   * @param configurations the list of grammatical configurations, sorted by descending score
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   *
   * @return the valid grammatical configurations for the given attachment
   */
  override fun getValidConfigurations(configurations: List<ScoredGrammar>,
                                      sentence: ParsingSentence,
                                      tokenIndex: Int,
                                      headIndex: Int?): List<ScoredGrammar> {

    val possibleMorphologies: List<Morphology> =
      sentence.tokens[tokenIndex].morphologies + sentence.getTokenMultiWordsMorphologies(tokenIndex)

    val correctDirection: SyntacticDependency.Direction =
      getDependencyDirection(tokenIndex = tokenIndex, headIndex = headIndex)
    val possibleDeprels: List<ScoredGrammar> = configurations.filter { it.config.direction == correctDirection }
    val worstScore: Double = configurations.last().score

    return if (possibleMorphologies.isNotEmpty())
      possibleDeprels.filter { it.config.isValid(possibleMorphologies) }.notEmptyOr {
        listOf(
          possibleMorphologies.first().buildGrammar(tokenIndex = tokenIndex, headIndex = headIndex, score = worstScore))
      }
    else
      possibleDeprels.filter { it.config.isSingleContentWord() }.notEmptyOr {
        listOf(ScoredGrammar(
          config = GrammaticalConfiguration(GrammaticalConfiguration.Component(
            SyntacticDependency(annotation = UNKNOWN_LABEL, direction = correctDirection))),
          score = worstScore))
      }
  }

  /**
   * Get the morphologies that are compatible with the deprel of a given token.
   *
   * @param token a parsing token
   * @param morphologies the list of possible morphologies of the token
   * @param grammaticalConfiguration the grammatical configuration of the token
   *
   * @return the morphologies compatible with the given deprel
   */
  override fun getValidMorphologies(token: ParsingToken,
                                    morphologies: List<Morphology>,
                                    grammaticalConfiguration: GrammaticalConfiguration): List<Morphology> {

    val posList: List<POSTag.Base> = grammaticalConfiguration.components.mapNotNull { it.pos as? POSTag.Base }

    return when {
      posList.isNotEmpty() -> morphologies.filter { it.components.map { it.pos.baseAnnotation } == posList }
      posList.first().type != POS.Num -> // In this case the configuration always defines a single morphology.
        listOf(Morphology(
          morphologies = listOf(SingleMorphology(
            lemma = token.form,
            pos = posList.first().type,
            allowIncompleteProperties = true))
        ))
      else -> listOf() // TODO: handle Number morphology
    }
  }

  /**
   * Return the morphologies associated to the multi-words that involve a given token.
   *
   * @param tokenIndex the index of the token
   *
   * @return a list of morphologies
   */
  private fun ParsingSentence.getTokenMultiWordsMorphologies(tokenIndex: Int): List<Morphology> =
    this.multiWords
      .filter { tokenIndex in it.startToken..it.endToken }
      .flatMap { it.morphologies }

  /**
   * @param possibleMorphologies the list of possible morphologies of the token to which this deprel is assigned
   *
   * @return whether this deprel is valid respect to the given morphologies
   */
  private fun GrammaticalConfiguration.isValid(possibleMorphologies: List<Morphology>): Boolean {

    val posTags: List<POSTag?> = this.components.map { it.pos }

    return possibleMorphologies.any { it.components.map { it.pos.baseAnnotation } == posTags }
  }

  /**
   * Build the deprel represented by this morphology annotation.
   *
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   * @param score the score to assign to the new deprel
   *
   * @return a new deprel represented in this morphology annotation
   */
  private fun Morphology.buildGrammar(tokenIndex: Int, headIndex: Int?, score: Double): ScoredGrammar {

    val direction: SyntacticDependency.Direction =
      getDependencyDirection(tokenIndex = tokenIndex, headIndex = headIndex)

    return ScoredGrammar(
      config = GrammaticalConfiguration(components = *Array(
        size = this.components.size,
        init = { i -> GrammaticalConfiguration.Component(SyntacticDependency(
          annotation = if (i == 0) "TOP" else "UNKNOWN",
          direction = direction))
        })
      ),
      score = score
    )
  }

  /**
   * @return whether this deprel defines a content word with a single morphology
   */
  private fun GrammaticalConfiguration.isSingleContentWord(): Boolean {

    val posTags: List<POSTag.Base> = this.components.map { it.pos as POSTag.Base }

    return posTags.size == 1 && posTags.first().type.isContentWord
  }
}
