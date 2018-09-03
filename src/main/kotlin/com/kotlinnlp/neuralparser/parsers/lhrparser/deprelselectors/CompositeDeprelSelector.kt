/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.MorphologyFactory
import com.kotlinnlp.linguisticdescription.morphology.MorphologyType
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.ContentWord
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
     * Morphology types associated by base annotation.
     */
    private val morphologiesByBaseAnnotation: Map<String, MorphologyType> =
      MorphologyType.values().filter { it.annotation == it.baseAnnotation }.associateBy { it.baseAnnotation }

    /**
     * @param posTag a POS tag
     *
     * @return the base morphology type associated to the given POS tag
     */
    private fun getMorphologyType(posTag: String): MorphologyType = this.morphologiesByBaseAnnotation.getValue(posTag)

    /**
     * Get the position of a deprel.
     *
     * @param tokenIndex the index of the token to which the deprel must be assigned
     * @param headIndex the index of the token head (can be null)
     *
     * @return the deprel position related to the given dependency relation
     */
    private fun getDeprelPosition(tokenIndex: Int, headIndex: Int?): Deprel.Position =
      headIndex?.let { if (tokenIndex < headIndex) Deprel.Position.LEFT else Deprel.Position.RIGHT }
        ?: Deprel.Position.ROOT
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

    val correctPosition: Deprel.Position = getDeprelPosition(tokenIndex = tokenIndex, headIndex = headIndex)
    val possibleDeprels: List<ScoredDeprel> = deprels.filter { it.value.direction == correctPosition }
    val worstScore: Double = deprels.last().score

    return if (possibleMorphologies.isNotEmpty())
      possibleDeprels.filter { it.value.isValid(possibleMorphologies) }.notEmptyOr {
        listOf(
          possibleMorphologies.first().buildDeprel(tokenIndex = tokenIndex, headIndex = headIndex, score = worstScore))
      }
    else
      possibleDeprels.filter { it.value.isSingleContentWord() }.notEmptyOr {
        listOf(ScoredDeprel(value = Deprel(label = UNKNOWN_LABEL, direction = correctPosition), score = worstScore))
      }
  }

  /**
   * Get the morphologies that are compatible with the deprel of a given token.
   *
   * @param token a parsing token
   * @param morphologies the list of possible morphologies of the token
   * @param deprelLabel the deprel label of the token
   *
   * @return the morphologies compatible with the given deprel
   */
  override fun getValidMorphologies(token: ParsingToken,
                                    morphologies: List<Morphology>,
                                    deprelLabel: String): List<Morphology> {

    val posTags: List<String> = deprelLabel.extractPosTags()
    val possibleMorphologies: List<Morphology> = morphologies.filter {
      it.list.map { it.type.baseAnnotation } == posTags
    }

    return if (possibleMorphologies.isNotEmpty())
      possibleMorphologies
    else
    // In this case the deprel label always defines a single morphology.
      listOf(Morphology(
        morphologies = listOf(MorphologyFactory(
          lemma = token.form,
          type = getMorphologyType(posTags.first()),
          allowIncompleteProperties = true))
      ))
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

    return possibleMorphologies.any { it.list.map { it.type.baseAnnotation } == posTags }
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
      label = this.buildDeprelLabel(topToken = headIndex == null),
      direction = getDeprelPosition(tokenIndex = tokenIndex, headIndex = headIndex)),
    score = score
  )

  /**
   * Build a label for a unknown deprel using the POS tags of this morphology.
   *
   * @param topToken whether the token to which this deprel is associated is the top of the sentence
   *
   * @return a [Deprel] label
   */
  private fun Morphology.buildDeprelLabel(topToken: Boolean): String =
    this.list
      .mapIndexed { i, it -> it.type.baseAnnotation + "~" + if (i == 0 && topToken) "TOP" else "UNKNOWN" }
      .joinToString("+")

  /**
   * @return whether this deprel defines a content word with a single morphology
   */
  private fun Deprel.isSingleContentWord(): Boolean {

    val posTags: List<String> = this.label.extractPosTags()

    if (posTags.size == 1) {

      val posTag: String = posTags.first()

      if (posTag != "NUM") { // The NUM is not a content word and cannot be built with the MorphologyFactory.

        val morphology: SingleMorphology =
          MorphologyFactory(lemma = "", type = getMorphologyType(posTag), allowIncompleteProperties = true)

        if (morphology is ContentWord) return true
      }
    }

    return false
  }

  /**
   * @return the list of POS tags annotated in the label of this deprel
   */
  private fun String.extractPosTags(): List<String> = this.split("+").map { it.split("~").first() }
}
