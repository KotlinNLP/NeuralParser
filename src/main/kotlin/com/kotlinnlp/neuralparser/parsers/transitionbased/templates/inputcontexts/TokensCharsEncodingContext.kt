/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts

import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The tokens context with an encoding representations for each token, built with chars and word embeddings.
 *
 * @property items a list of items
 * @property sentence a parsing sentence, with the tokens list that is parallel to [items]
 * @property usedEncoders a list of HAN encoders used to encode tokens chars, one per token
 * @property charsEmbeddings a list of chars embeddings lists, one per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property encodingSize the size of the each encoding
 * @property nullItemVector used to represent the encoding of a null item of the decoding window
 * @property tokensEncodings a list of encoding representations of the tokens
 * @property trainingMode whether the parser is being trained
 */
class TokensCharsEncodingContext(
  override val items: List<DenseItem>,
  sentence: ParsingSentence,
  val usedEncoders: List<HANEncoder<DenseNDArray>>,
  val charsEmbeddings: List<List<Embedding>>,
  val wordEmbeddings: List<Embedding>,
  encodingSize: Int,
  nullItemVector: DenseNDArray,
  val tokensEncodings: List<DenseNDArray>,
  trainingMode: Boolean = false
) : TokensEncodingContext<TokensCharsEncodingContext>(
  sentence = sentence,
  encodingSize = encodingSize,
  nullItemVector = nullItemVector,
  trainingMode = trainingMode
) {

  /**
   * @param itemId the id of an item (can be null)
   *
   * @return get the encoding vector of the item with the given id or the [nullItemVector] if [itemId] is null
   */
  override fun getTokenEncoding(itemId: Int?): DenseNDArray
    = if (itemId != null) this.tokensEncodings[itemId] else this.nullItemVector

  /**
   * @return a copy of this [TokensCharsEncodingContext]
   */
  override fun copy() = TokensCharsEncodingContext(
    items = this.items,
    sentence = this.sentence,
    usedEncoders = this.usedEncoders,
    charsEmbeddings = this.charsEmbeddings,
    wordEmbeddings = this.wordEmbeddings,
    encodingSize = this.encodingSize,
    nullItemVector = this.nullItemVector,
    tokensEncodings = this.tokensEncodings,
    trainingMode = this.trainingMode
  )
}
