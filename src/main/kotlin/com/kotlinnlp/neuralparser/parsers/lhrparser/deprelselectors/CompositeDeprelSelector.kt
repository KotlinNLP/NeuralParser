/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils.ScoredDeprel

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
     * The unknown deprel, returned when no deprel can be selected.
     * TODO: replace with better heuristic
     */
    val UNKNOWN_DEPREL = Deprel(label = "NOUN~UNKNOWN", direction = Deprel.Position.NULL)
  }

  /**
   * Get the best deprel of a prediction.
   *
   * @param deprels the list of scored deprels
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   *
   * @return the best deprel
   */
  override fun getBestDeprel(deprels: List<ScoredDeprel>, sentence: ParsingSentence, tokenIndex: Int): Deprel {

    val possibleMorphologies: List<Morphology> =
      sentence.tokens[tokenIndex].morphologies + sentence.getTokenMultiWordsMorphologies(tokenIndex)

    return deprels.firstOrNull {

      val posTags: List<String> = it.value.label.extractPosTags()

      possibleMorphologies.any { it.list.map { it.type.baseAnnotation } == posTags }

    }?.value ?: UNKNOWN_DEPREL
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

    return morphologies.filter { it.list.map { it.type.baseAnnotation } == posTags }
  }

  /**
   * @return the list of POS tags annotated in the label of this deprel
   */
  private fun String.extractPosTags(): List<String> = this.split(" ").map { it.split("~").first() }

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
}
