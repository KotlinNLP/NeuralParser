/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.preprocessors

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.helpers.labelerselector.NoFilterSelector

/**
 * Pre-process a sentence that has been built from a [CoNLLSentence].
 *
 * @param conllSentences the list of CoNLL sentences from which the input base sentences are built
 */
class CoNLLPreprocessor(private val conllSentences: List<CoNLLSentence>) : SentencePreprocessor {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Convert a [BaseSentence] to a [ParsingSentence].
   *
   * @param sentence a base sentence
   *
   * @return a sentence ready to be parsed
   */
  override fun convert(sentence: BaseSentence): ParsingSentence {

    val conllTokens: List<CoNLLToken> = this.conllSentences[sentence.position.index].tokens

    return ParsingSentence(
      tokens = sentence.tokens.mapIndexed { i, it ->
        ParsingToken(
          id = it.id,
          form = it.form,
          position = it.position,
          pos = conllTokens[i].posList
        )
      },
      labelerSelector = NoFilterSelector,
      position = sentence.position
    )
  }
}
