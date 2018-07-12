/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The latent syntactic structure encoded by the TokensEncoder, the ContextEncoder and the HeadsEncoder.
 *
 * @property sentence the sentence containing the tokens
 * @property tokensEncoding the tokens encoded by the TokensEncoder
 * @property contextVectors the context vectors encoded by the ContextEncoder
 * @property latentHeads the latent heads encoded by the HeadsEncoder
 * @property virtualRoot the vector that represents the root token of a sentence
 */
data class LatentSyntacticStructure(
  val sentence: ParsingSentence,
  val tokensEncoding: List<DenseNDArray>,
  val contextVectors: List<DenseNDArray>,
  val latentHeads: List<DenseNDArray>,
  val virtualRoot: DenseNDArray
) {

  /**
   * The length of the sentence.
   */
  val size: Int = this.sentence.tokens.size
}