/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The tokens context with an encoding representation for each token, built using embeddings vectors of word and POS.
 *
 * @property items a list of items
 * @property tokens a list of tokens, parallel to [items]
 * @property posEmbeddings a list of pos embeddings, one per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property nullItemVector used to represent the encoding of a null item of the decoding window
 * @property encodingSize the size of the each encoding
 * @property tokensEncodings the size of the encodings
 * @property skipPunctuation whether to initialize a [State] without punctuation items
 */
class TokensEmbeddingsContext(
  override val items: List<DenseItem>,
  tokens: List<Token>,
  val posEmbeddings: List<Embedding>,
  val wordEmbeddings: List<Embedding>,
  unknownItemVector: DenseNDArray,
  encodingSize: Int,
  val tokensEncodings: List<DenseNDArray>,
  private val skipPunctuation: Boolean = false,
  trainingMode: Boolean = false
) : TokensEncodingContext<TokensEmbeddingsContext>(
  tokens = tokens,
  encodingSize = encodingSize,
  nullItemVector = unknownItemVector,
  trainingMode = trainingMode) {

  /**
   * @param itemId the id of an item
   *
   * @return get the encoding vector of the item with the given id
   */
  override fun getTokenEncoding(itemId: Int?): DenseNDArray
    = if (itemId != null) this.tokensEncodings[itemId] else this.nullItemVector

  /**
   * @return the items id to initialize a [State]
   */
  override fun getInitialStateItemsId() = if (!this.skipPunctuation)
    super.getInitialStateItemsId()
  else
    this.tokens
    .filter { !it.isPunctuation }
    .map { it.id }

  /**
   * @return a copy of this [TokensEmbeddingsContext]
   */
  override fun copy() = TokensEmbeddingsContext(
    items = this.items,
    tokens = this.tokens,
    posEmbeddings = this.posEmbeddings,
    wordEmbeddings = this.wordEmbeddings,
    unknownItemVector = this.nullItemVector,
    encodingSize = this.encodingSize,
    tokensEncodings = this.tokensEncodings,
    trainingMode = this.trainingMode)
}
