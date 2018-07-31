/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils.ScoredDeprel

/**
 * The selector of the valid morphologies and the best deprel of a prediction.
 */
interface MorphoDeprelSelector {

  /**
   * Get the best deprel of a prediction.
   *
   * @param deprels the list of scored deprels
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   *
   * @return the best deprel
   */
  fun getBestDeprel(deprels: List<ScoredDeprel>, sentence: ParsingSentence, tokenIndex: Int): Deprel

  /**
   * Get the morphologies that are compatible with the deprel of a token.
   *
   * @param morphologies the list of possible morphologies of the token
   * @param deprelLabel the deprel label of the token
   *
   * @return the morphologies compatible with the given deprel
   */
  fun getValidMorphologies(morphologies: List<Morphology>, deprelLabel: String): List<Morphology>
}
