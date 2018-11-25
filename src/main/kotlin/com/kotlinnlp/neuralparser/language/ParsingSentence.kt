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

/**
 * The sentence used as input of the [com.kotlinnlp.neuralparser.NeuralParser].
 *
 * @property tokens the list of tokens of the sentence
 * @property morphoAnalysis the morphological analysis of the tokens (can be null)
 */
class ParsingSentence(
  override val tokens: List<ParsingToken>,
  override val morphoAnalysis: MorphologicalAnalysis? = null
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
}
