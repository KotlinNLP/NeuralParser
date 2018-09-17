/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.Deprel
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel
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
     * Get the direction of a deprel.
     *
     * @param tokenIndex the index of the token to which the deprel must be assigned
     * @param headIndex the index of the token head (can be null)
     *
     * @return the deprel direction related to the given grammatical configuration
     */
    private fun getDeprelDirection(tokenIndex: Int, headIndex: Int?): Deprel.Direction =
      headIndex?.let { if (tokenIndex < headIndex) Deprel.Direction.LEFT else Deprel.Direction.RIGHT }
        ?: Deprel.Direction.ROOT
  }

  /**
   * Get the list of deprels that are valid for a given attachment.
   *
   * @param deprels the list of deprels, sorted by descending score
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   *
   * @return the valid deprels for the given attachment
   */
  override fun getValidDeprels(deprels: List<ScoredDeprel>,
                               sentence: ParsingSentence,
                               tokenIndex: Int,
                               headIndex: Int?): List<ScoredDeprel> {

    val possibleMorphologies: List<Morphology> =
      sentence.tokens[tokenIndex].morphologies + sentence.getTokenMultiWordsMorphologies(tokenIndex)

    val correctDirection: Deprel.Direction = getDeprelDirection(tokenIndex = tokenIndex, headIndex = headIndex)
    val possibleDeprels: List<ScoredDeprel> = deprels.filter { it.value.direction == correctDirection }
    val worstScore: Double = deprels.last().score

    return if (possibleMorphologies.isNotEmpty())
      possibleDeprels.filter { it.value.isValid(possibleMorphologies) }.notEmptyOr {
        listOf(
          possibleMorphologies.first().buildDeprel(tokenIndex = tokenIndex, headIndex = headIndex, score = worstScore))
      }
    else
      possibleDeprels.filter { it.value.isSingleContentWord() }.notEmptyOr {
        listOf(ScoredDeprel(
          value = Deprel(labels = listOf(UNKNOWN_LABEL), direction = correctDirection),
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

    val posTags: List<String> = grammaticalConfiguration.posTag?.labels ?: listOf()
    val possibleMorphologies: List<Morphology> = morphologies.filter {
      it.list.map { it.pos.baseAnnotation } == posTags
    }

    return when {
      possibleMorphologies.isNotEmpty() -> possibleMorphologies
      posTags.first() != POS.Num.baseAnnotation -> // In this case the deprel label always defines a single morphology.
        listOf(Morphology(
          morphologies = listOf(SingleMorphology(
            lemma = token.form,
            pos = POS.byBaseAnnotation(posTags.first()),
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
  private fun Deprel.isValid(possibleMorphologies: List<Morphology>): Boolean {

    val posTags: List<String> = this.label.extractPosTags()

    return possibleMorphologies.any { it.list.map { it.pos.baseAnnotation } == posTags }
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
  private fun Morphology.buildDeprel(tokenIndex: Int, headIndex: Int?, score: Double) = ScoredDeprel(
    value = Deprel(
      labels = listOf("TOP") + (1 until this.list.size).map { "UNKNOWN" },
      direction = getDeprelDirection(tokenIndex = tokenIndex, headIndex = headIndex)),
    score = score
  )

  /**
   * @return whether this deprel defines a content word with a single morphology
   */
  private fun Deprel.isSingleContentWord(): Boolean {

    val posTags: List<String> = this.label.extractPosTags()

    return posTags.size == 1 && POS.byBaseAnnotation(posTags.first()).isContentWord
  }
}
