/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.sentenceconverters

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.tokensencoder.wrapper.SentenceConverter

/**
 * The sentence converter that mirrors the input sentence to the output.
 */
class MirrorConverter<TokenType: Token, SentenceType: Sentence<TokenType>>
  : SentenceConverter<TokenType, SentenceType, TokenType, SentenceType> {

  /**
   * Return the same sentence given in input.
   *
   * @param sentence the input sentence
   *
   * @return the same sentence given in input
   */
  override fun convert(sentence: SentenceType): SentenceType = sentence
}
