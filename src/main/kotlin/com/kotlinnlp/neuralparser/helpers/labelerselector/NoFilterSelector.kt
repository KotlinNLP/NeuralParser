/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.labelerselector

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar

/**
 * The selector that does not filter.
 */
object NoFilterSelector : LabelerSelector {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

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

    val correctDirection = SyntacticDependency.Direction(tokenIndex = tokenIndex, headIndex = headIndex)

    return configurations.filter { it.config.direction == correctDirection }
  }

  /**
   * Return all the morphologies as valid.
   *
   * @param sentence the input sentence
   * @param tokenIndex the index of a token of the sentence
   * @param configuration the grammatical configuration of the token
   *
   * @return all the given morphologies
   */
  override fun getValidMorphologies(sentence: ParsingSentence,
                                    tokenIndex: Int,
                                    configuration: GrammaticalConfiguration): List<Morphology> =
    sentence.morphoAnalysis!!.allMorphologies[tokenIndex].filter {
      it.components.size == configuration.components.size
    }
}
