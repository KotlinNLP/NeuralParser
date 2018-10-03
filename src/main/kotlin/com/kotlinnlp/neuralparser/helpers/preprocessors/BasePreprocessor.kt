/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.preprocessors

import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken

/**
 * Pre-process a sentence before the parsing starts.
 */
class BasePreprocessor : SentencePreprocessor {

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
  override fun convert(sentence: BaseSentence) = ParsingSentence(
    tokens = sentence.tokens.map {
      ParsingToken(id = it.id, form = it.form, position = it.position, morphologies = emptyList())
    }
  )
}

