/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.lssencoder.LSSEncoder
import com.kotlinnlp.lssencoder.tokensencoder.LSSTokensEncoder
import com.kotlinnlp.morphodisambiguator.helpers.dataset.Dataset
import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.morphodisambiguator.language.MorphoToken
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.sequenceencoder.SequenceParallelEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

class MorphoDisambiguator (val model: MorphoDisambiguatorModel){

  /**
   * The tokens encoder
   */
  private val tokensEncoder = LSSTokensEncoder(this.model.lssModel, useDropout = false)

  /**
   * The BIRNN encoder on top of the Lss Encoder
   */
  private val biRNNEncoder = BiRNNEncoder<DenseNDArray>(
      network = this.model.biRNNEncoderModel,
      propagateToInput = true,
      useDropout = false)

  /**
   * The sequence parallel encoder on top of biRNN encoder
   */
  private val outputEncoder = SequenceParallelEncoder<DenseNDArray> (
      model = this.model.sequenceParallelEncoderModel,
      propagateToInput = true,
      useDropout = false)

  /**
   * @param sentence: The sentence to disambiguate
   *
   * @return a list of [MorphoToken]
   */
  fun disambiguate(sentence: ParsingSentence) : List<MorphoToken> {
    val disambiguatedTokens: List<MorphoToken> = mutableListOf()
    return disambiguatedTokens
  }


}