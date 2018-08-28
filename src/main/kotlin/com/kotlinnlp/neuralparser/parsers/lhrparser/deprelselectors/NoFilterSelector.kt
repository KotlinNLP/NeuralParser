/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel

/**
 * The selector that does not filter.
 */
class NoFilterSelector : MorphoDeprelSelector {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Get the list of deprels that are valid for a given attachment.
   *
   * @param deprels the list of scored deprels
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   *
   * @return the valid deprels for the given attachment
   */
  override fun getValidDeprels(deprels: List<ScoredDeprel>,
                               sentence: ParsingSentence,
                               tokenIndex: Int,
                               headIndex: Int?): List<ScoredDeprel> = deprels

  /**
   * Return all the morphologies as valid.
   *
   * @param token a parsing token
   * @param morphologies the list of possible morphologies of the token
   * @param deprelLabel the deprel label of the token
   *
   * @return all the given morphologies
   */
  override fun getValidMorphologies(token: ParsingToken,
                                    morphologies: List<Morphology>,
                                    deprelLabel: String): List<Morphology> = morphologies
}
