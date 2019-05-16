/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.preprocessors

import com.kotlinnlp.linguisticdescription.morphology.MorphologicalAnalysis
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.helpers.labelerselector.MorphoSelector

/**
 * Pre-process a sentence with a morphological analyzer, before the parsing starts.
 *
 * @param dictionary a morphologies dictionary
 */
class MorphoPreprocessor(private val dictionary: MorphologyDictionary) : SentencePreprocessor {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * A morphological analyzer as transient property.
   */
  @kotlin.jvm.Transient private var morphologicalAnalyzer: MorphologicalAnalyzer? = null

  /**
   * Convert a [BaseSentence] to a [ParsingSentence].
   *
   * @param sentence a base sentence
   *
   * @return a sentence ready to be parsed
   */
  override fun convert(sentence: BaseSentence): ParsingSentence {

    @Suppress("UNCHECKED_CAST")
    val morphoAnalysis: MorphologicalAnalysis = this.getOrInitAnalyzer().analyze(sentence as RealSentence<RealToken>)

    return ParsingSentence(
      tokens = sentence.tokens.map {
        ParsingToken(
          id = it.id,
          form = it.form,
          position = it.position
        )
      },
      morphoAnalysis = morphoAnalysis,
      labelerSelector = MorphoSelector,
      position = sentence.position
    )
  }

  /**
   * Get the [MorphologicalAnalyzer] of this preprocessor, eventually initializing it (in case this class has just been
   * deserialized).
   *
   * @return the morphological analyzer of this preprocessor
   */
  private fun getOrInitAnalyzer(): MorphologicalAnalyzer {

    if (this.morphologicalAnalyzer == null)
      this.morphologicalAnalyzer = MorphologicalAnalyzer(this.dictionary)

    return this.morphologicalAnalyzer!!
  }
}
