/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * A base real sentence.
 *
 * @property tokens the list of tokens that compose the sentence
 * @property position the position of this sentence in the original text
 */
data class BaseSentence(
  override val tokens: List<RealToken>,
  override val position: Position
) : RealSentence<RealToken> {

  companion object {

    /**
     * Convert a CoNLL sentence to a [BaseSentence].
     *
     * @param sentence a CoNLL sentence
     *
     * @return a real sentence of real tokens
     */
    fun fromCoNLL(sentence: CoNLLSentence): BaseSentence {

      val baseTokens = sentence.tokens.toBaseTokens()

      return BaseSentence(
        tokens = baseTokens,
        position = Position(index = 0, start = baseTokens.first().position.start, end = baseTokens.last().position.end))
    }
  }
}
