/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts

import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The tokens context with an encoding representation for each token, built using embeddings vectors of word and POS.
 *
 * @property items a list of items
 * @property sentence a parsing sentence, with the tokens list that is parallel to [items]
 * @property posEmbeddings a list of pos embeddings, one per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property nullItemVector used to represent the encoding of a null item of the decoding window
 * @property encodingSize the size of the each encoding
 * @property tokensEncodings the list of tokens encodings
 * @property skipPunctuation whether to initialize a [State] without punctuation items
 * @property trainingMode whether the parser is being trained
 */
class TokensEmbeddingsContext(
  sentence: ParsingSentence,
  val posEmbeddings: List<Embedding>,
  val wordEmbeddings: List<Embedding>,
  nullItemVector: DenseNDArray,
  encodingSize: Int,
  tokensEncodings: List<DenseNDArray>,
  private val skipPunctuation: Boolean = false,
  trainingMode: Boolean = false
) : TokensEncodingContext<TokensEmbeddingsContext>(
  sentence = sentence,
  encodingSize = encodingSize,
  tokensEncodings = tokensEncodings,
  nullItemVector = nullItemVector,
  trainingMode = trainingMode
) {

  /**
   * @return the item ids used to initialize a [State]
   */
  override fun getInitialStateItemIds() = if (!this.skipPunctuation)
    super.getInitialStateItemIds()
  else
    this.sentence.tokens
    .filter { !it.isPunctuation }
    .map { it.id }

  /**
   * @return a copy of this [TokensEmbeddingsContext]
   */
  override fun copy() = TokensEmbeddingsContext(
    sentence = this.sentence,
    posEmbeddings = this.posEmbeddings,
    wordEmbeddings = this.wordEmbeddings,
    nullItemVector = this.nullItemVector,
    encodingSize = this.encodingSize,
    tokensEncodings = this.tokensEncodings,
    trainingMode = this.trainingMode)
}
