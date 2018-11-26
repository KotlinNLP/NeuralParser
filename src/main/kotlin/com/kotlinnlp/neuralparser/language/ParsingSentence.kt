/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.morphology.MorphologicalAnalysis
import com.kotlinnlp.linguisticdescription.morphology.Morphologies
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.neuralparser.helpers.labelerselector.LabelerSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar

/**
 * The sentence used as input of the [com.kotlinnlp.neuralparser.NeuralParser].
 *
 * @property tokens the list of tokens of the sentence
 * @property morphoAnalysis the morphological analysis of the tokens (can be null)
 * @param labelerSelector the labeler selector used to select the grammatical configurations compatible with the sentence
 */
class ParsingSentence(
  override val tokens: List<ParsingToken>,
  override val morphoAnalysis: MorphologicalAnalysis? = null,
  private val labelerSelector: LabelerSelector
) : MorphoSentence<ParsingToken>, SentenceIdentificable<ParsingToken>() {

  /**
   * Check whether the morphologies of the token are compatible with the given configuration [c].
   * Middle multi-words morphologies are compared partially (only with the "CONTIN" components).
   *
   * @param c the grammatical configuration
   * @param tokenIndex the index of a token of the sentence
   *
   * @return true if the morphologies of the token are compatible with the given configuration, otherwise false
   */
  fun areConfigurationCompatible(c: GrammaticalConfiguration, tokenIndex: Int): Boolean =
    this.morphoAnalysis!!.startMorphologies[tokenIndex].any { c.isCompatible(it) } ||
      this.morphoAnalysis.middleMWMorphologies[tokenIndex].any { c.isPartiallyCompatible(it) }

  /**
   * @param c the grammatical configuration
   * @param tokenIndex the index of a token of the sentence
   *
   * @return the token morphologies (including the multi-words) that are compatible with the given configuration
   */
  fun getCompatibleMorphologies(c: GrammaticalConfiguration, tokenIndex: Int) = Morphologies(
    this.morphoAnalysis!!.allMorphologies[tokenIndex].filter {
      c.isCompatible(it) // TODO: || c.isPartiallyCompatible(it)
    })

  /**
   * Get the list of scored grammatical configurations that are valid for a given attachment.
   *
   * @param tokenIndex the index of the token to which one of the [configurations] must be assigned
   * @param headIndex the index of the token head (can be null)
   * @param configurations the list of grammatical configurations, sorted by descending score
   *
   * @return the valid grammatical configurations for the given attachment
   */
  fun getValidConfigurations(tokenIndex: Int,
                             headIndex: Int?,
                             configurations: List<ScoredGrammar>): List<ScoredGrammar> =
    this.labelerSelector.getValidConfigurations(
      sentence = this,
      tokenIndex = tokenIndex,
      headIndex = headIndex,
      configurations = configurations)

  /**
   * Get the morphologies of a given token that are compatible with the given grammatical configuration.
   *
   * @param tokenIndex the index of a token of the sentence
   * @param configuration the grammatical configuration of the token
   *
   * @return the morphologies compatible with the given grammatical configuration
   */
  fun getValidMorphologies(tokenIndex: Int,
                           configuration: GrammaticalConfiguration): Morphologies =
    this.labelerSelector.getValidMorphologies(
      sentence = this,
      tokenIndex = tokenIndex,
      configuration = configuration)
}
