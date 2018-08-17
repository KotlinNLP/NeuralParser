/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.core.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The tokens context with an encoding representation for each token, built using an embedding vector of the word and
 * a binary vector of the ambiguous POS tags.
 *
 * @property sentence a parsing sentence, with the tokens list that is parallel to [items]
 * @property posTags a list of ambiguous POS tags, one list per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property preTrainedWordEmbeddings a list of pre-trained word embeddings, one per token (can be null=
 * @property nullItemVector used to represent the encoding of a null item of the decoding window
 * @property encodingSize the size of the each encoding
 * @property tokensEncodings the list of tokens encodings
 * @property trainingMode whether the parser is being trained
 */
class TokensAmbiguousPOSContext(
  sentence: ParsingSentence,
  val posTags: List<List<POSTag>>,
  val wordEmbeddings: List<Embedding>,
  val preTrainedWordEmbeddings: List<Embedding>?,
  nullItemVector: DenseNDArray,
  encodingSize: Int,
  tokensEncodings: List<DenseNDArray>,
  trainingMode: Boolean
) : TokensEncodingContext<TokensAmbiguousPOSContext>(
  sentence = sentence,
  encodingSize = encodingSize,
  tokensEncodings = tokensEncodings,
  nullItemVector = nullItemVector,
  trainingMode = trainingMode
) {

  /**
   * @return a copy of this [TokensAmbiguousPOSContext]
   */
  override fun copy() = TokensAmbiguousPOSContext(
    sentence = this.sentence,
    posTags = this.posTags,
    wordEmbeddings = this.wordEmbeddings,
    preTrainedWordEmbeddings = this.preTrainedWordEmbeddings,
    nullItemVector = this.nullItemVector,
    encodingSize = this.encodingSize,
    tokensEncodings = this.tokensEncodings,
    trainingMode = this.trainingMode)
}
