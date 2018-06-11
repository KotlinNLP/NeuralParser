/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.deeplearning.treernn.TreeEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The tokens context with an encoding representation for each token.
 *
 * @property items a list of items
 * @property tokens a list of tokens, parallel to [items]
 * @property posEmbeddings a list of pos embeddings, one per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property nullItemVector used to represent the encoding of a null item of the decoding window
 * @property encodingSize the size of the each encoding
 * @property tokensEncodings the size of the encodings
 */
class TreeEncoderContext(
  override val items: List<DenseItem>,
  tokens: List<Token>,
  val posEmbeddings: List<Embedding>,
  val wordEmbeddings: List<Embedding>,
  nullItemVector: DenseNDArray,
  encodingSize: Int,
  val tokensEncodings: List<TreeEncoder.Node>,
  trainingMode: Boolean
) : TokensEncodingContext<TreeEncoderContext>(
  tokens = tokens,
  nullItemVector = nullItemVector,
  encodingSize = encodingSize,
  trainingMode = trainingMode
) {

  /**
   * @param itemId the id of an item
   *
   * @return get the encoding vector of the item with the given id
   */
  override fun getTokenEncoding(itemId: Int?): DenseNDArray
    = if (itemId != null) this.tokensEncodings[itemId].encoding else this.nullItemVector

  /**
   * @return a copy of this [TreeEncoderContext]
   */
  override fun copy(): TreeEncoderContext {
    TODO("not implemented")
  }
}
