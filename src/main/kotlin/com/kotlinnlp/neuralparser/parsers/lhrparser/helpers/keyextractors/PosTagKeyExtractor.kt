/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.keyextractors

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingKeyExtractor

/**
 * An [EmbeddingKeyExtractor] by POS tag.
 */
object PosTagKeyExtractor : EmbeddingKeyExtractor<ParsingToken, Sentence<ParsingToken>> {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * @param sentence a generic sentence
   * @param tokenId the id of the token from which to extract the key
   *
   * @return the POS as string
   */
  override fun getKey(sentence: Sentence<ParsingToken>, tokenId: Int): String =
    sentence.tokens[tokenId].pos?.toString() ?: "_"
}
