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
 * A base real sentence.
 *
 * @property id the id of the sentence, unique within a list of sentences
 * @property tokens the list of tokens that compose the sentence
 * @property position the position of this sentence in the original text
 */
data class BaseSentence(
  val id: Int,
  override val tokens: List<BaseToken>,
  override val position: Position
) : RealSentence<BaseToken> {

  companion object {

    /**
     * Convert a CoNLL sentence to a [BaseSentence].
     *
     * @param sentence a CoNLL sentence
     * @param index the index of the sentence within a list of sentences
     *
     * @return a real sentence of real tokens
     */
    fun fromCoNLL(sentence: CoNLLSentence, index: Int): BaseSentence {

      val baseTokens = sentence.tokens.toBaseTokens()

      return BaseSentence(
        id = index, // the index is unique within a list of sentences
        tokens = baseTokens,
        position = Position(
          index = index,
          start = baseTokens.first().position.start,
          end = baseTokens.last().position.end))
    }
  }
}
