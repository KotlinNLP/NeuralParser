/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.linguisticdescription.DependencyRelation
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel
import java.io.Serializable

/**
 * The selector of the valid morphologies and the best deprel of a prediction.
 */
interface MorphoDeprelSelector : Serializable {

  /**
   * Get the list of deprels that are valid for a given attachment.
   *
   * @param deprels the list of deprels, sorted by descending score
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   *
   * @return the valid deprels for the given attachment
   */
  fun getValidDeprels(deprels: List<ScoredDeprel>,
                      sentence: ParsingSentence,
                      tokenIndex: Int,
                      headIndex: Int?): List<ScoredDeprel>

  /**
   * Get the morphologies that are compatible with the deprel of a given token.
   *
   * @param token a parsing token
   * @param morphologies the list of possible morphologies of the token
   * @param dependencyRelation the dependency relation of the token
   *
   * @return the morphologies compatible with the given dependency relation
   */
  fun getValidMorphologies(token: ParsingToken,
                           morphologies: List<Morphology>,
                           dependencyRelation: DependencyRelation): List<Morphology>
}
