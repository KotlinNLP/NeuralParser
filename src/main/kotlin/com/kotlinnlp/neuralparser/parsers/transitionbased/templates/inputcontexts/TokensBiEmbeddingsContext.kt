/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts

import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.utils.items.MultiDenseItem
import com.kotlinnlp.simplednn.deeplearning.embeddings.Embedding
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.context.InputContext

/**
 * The tokens context with two encoding representations for each token.
 *
 * @property items a list of items
 * @property tokens a list of tokens, parallel to [items]
 * @property posEmbeddings a list of pos embeddings, one per token
 * @property wordEmbeddings a list of word embeddings, one per token
 * @property tokensDependentEncodings a list of encoding representations of the tokens, as dependents
 * @property tokensGovernorEncodings a list of encoding representations of the tokens, as governors
 */
class TokensBiEmbeddingsContext(
  override val items: List<MultiDenseItem>,
  val tokens: List<Token>,
  val posEmbeddings: List<Embedding>,
  val wordEmbeddings: List<Embedding>,
  val tokensDependentEncodings: List<DenseNDArray>,
  val tokensGovernorEncodings: List<DenseNDArray>,
  val trainingMode: Boolean = false
) :
  InputContext<TokensBiEmbeddingsContext, MultiDenseItem> {

  /**
   * The length of the sentence.
   */
  override val length: Int = this.tokens.size

  /**
   * @return a copy of this [TokensBiEmbeddingsContext]
   */
  override fun copy(): TokensBiEmbeddingsContext {
    TODO("not implemented")
  }
}
