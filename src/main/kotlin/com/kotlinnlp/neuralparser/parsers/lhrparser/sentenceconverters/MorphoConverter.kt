/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.sentenceconverters

import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.tokensencoder.wrapper.SentenceConverter

/**
 * The sentence converter from a [ParsingSentence] to a [MorphoSentence].
 */
class MorphoConverter : SentenceConverter<ParsingToken, ParsingSentence, FormToken, MorphoSentence<FormToken>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Convert a given [ParsingSentence] to a [MorphoSentence] simply casting it.
   *
   * @param sentence the input sentence
   *
   * @return the converted sentence
   */
  @Suppress("UNCHECKED_CAST")
  override fun convert(sentence: ParsingSentence): MorphoSentence<FormToken> = sentence as MorphoSentence<FormToken>
}
