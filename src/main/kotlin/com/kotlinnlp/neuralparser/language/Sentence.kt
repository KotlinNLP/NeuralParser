/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * Data class containing the properties of a sentence.
 *
 * @property tokens the list of tokens that compose the sentence
 * @property position the position of this sentence in the original text
 */
data class Sentence(
  override val tokens: List<Token>,
  override val position: Position
) : RealSentence<Token> {

  companion object {

    /**
     * Transform a CoNLL sentence into a [Sentence].
     *
     * @param sentence a CoNLL sentence
     * @return a basic sentence
     */
    fun fromCoNLL(sentence: CoNLLSentence): Sentence {

      val baseTokens = sentence.tokens.toTokens()

      return Sentence(
        tokens = baseTokens,
        position = Position(
          index = 0,
          start = baseTokens.first().position.start,
          end = baseTokens.last().position.end))
    }
  }
}