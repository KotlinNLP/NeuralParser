/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * @param id the token id, an incremental integer starting from 0 within a sentence
 * @param form the form
 * @param morphologies the list of possible morphologies of the token
 * @param posTag the best part-of-speech tags associated to the token (can be null) TODO: find a better solution
 */
data class ParsingToken(
  val id: Int,
  override val form: String,
  val morphologies: List<Morphology> = emptyList(),
  val posTag: String?
) : FormToken {

  /**
   * @param dependencyRelation the dependency relation of this token
   * @param position the position of the token in the sentence
   *
   * @return a new morpho syntactic token
   */
  fun toMorphoSyntacticToken(dependencyRelation: DependencyRelation, position: Position): MorphoSyntacticToken = Word(
    id = this.id,
    form = this.form,
    position = position,
    morphologies = emptyList(), // TODO: set it
    dependencyRelation = dependencyRelation)
}
