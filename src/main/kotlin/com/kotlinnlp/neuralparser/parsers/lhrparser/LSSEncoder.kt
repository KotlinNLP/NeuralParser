/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder.HeadsEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapper

/**
 * The main encoder that builds the latent syntactic structure.
 *
 * @property tokensEncoderWrapper the tokens encoder wrapped with a sentence converter from the [ParsingSentence]
 * @property contextEncoder the encoder of tokens encodings sentential context
 * @property headsEncoder the encoder that generated the latent heads representation
 * @property virtualRoot the vector that represents the root token of a sentence
 */
class LSSEncoder<TokenType: Token, SentenceType: Sentence<TokenType>>(
  val tokensEncoderWrapper: TokensEncoderWrapper<ParsingToken, ParsingSentence, TokenType, SentenceType>,
  val contextEncoder: ContextEncoder,
  val headsEncoder: HeadsEncoder,
  val virtualRoot: DenseNDArray
) {

  /**
   * @param sentence the sentence to encode
   *
   * @return the latent syntactic structure
   */
  fun encode(sentence: ParsingSentence): LatentSyntacticStructure {

    val tokensEncodings: List<DenseNDArray> = this.tokensEncoderWrapper.forward(sentence)
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings)
    val latentHeads: List<DenseNDArray> = this.headsEncoder.forward(contextVectors)

    return LatentSyntacticStructure(
      sentence = sentence,
      tokensEncoding = tokensEncodings,
      contextVectors = contextVectors,
      latentHeads = latentHeads,
      virtualRoot = this.virtualRoot)
  }
}
