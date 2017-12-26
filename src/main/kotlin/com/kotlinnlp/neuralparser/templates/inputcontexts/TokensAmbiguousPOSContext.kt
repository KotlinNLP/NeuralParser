/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.inputcontexts

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.deeplearning.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The tokens context with an encoding representation for each token, built using an embedding vector of the word and
 * a binary vector of the ambiguous POS tags.
 *
 * @property items a list of items
 * @property tokens a list of tokens, parallel to [items]
 * @property posTags a list of ambiguous POS tags, one list per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property preTrainedWordEmbeddings a list of pre-trained word embeddings, one per token (can be null=
 * @property nullItemVector used to represent the encoding of a null item of the decoding window
 * @property encodingSize the size of the each encoding
 * @property tokensEncodings the size of the encodings
 */
class TokensAmbiguousPOSContext(
  override val items: List<DenseItem>,
  tokens: List<Token>,
  val posTags: List<List<POSTag>>,
  val wordEmbeddings: List<Embedding>,
  val preTrainedWordEmbeddings: List<Embedding>?,
  unknownItemVector: DenseNDArray,
  encodingSize: Int,
  val tokensEncodings: Array<DenseNDArray>,
  trainingMode: Boolean
) : TokensEncodingContext<TokensAmbiguousPOSContext>(
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
   * @return a copy of this [TokensAmbiguousPOSContext]
   */
  override fun copy() = TokensAmbiguousPOSContext(
    items = this.items,
    tokens = this.tokens,
    posTags = this.posTags,
    wordEmbeddings = this.wordEmbeddings,
    preTrainedWordEmbeddings = this.preTrainedWordEmbeddings,
    unknownItemVector = this.nullItemVector,
    encodingSize = this.encodingSize,
    tokensEncodings = this.tokensEncodings,
    trainingMode = this.trainingMode)
}
