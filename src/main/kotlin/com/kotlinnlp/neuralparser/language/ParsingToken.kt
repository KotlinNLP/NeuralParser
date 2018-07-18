/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * @param id the token id
 * @param form the form
 * @param position the position of the token in its sentence
 * @param morphologies the list of possible morphologies of the token
 * @param posTag the best part-of-speech tags associated to the token (can be null) TODO: find a better solution
 */
data class ParsingToken(
  val id: Int,
  override val form: String,
  override val position: Position,
  val morphologies: List<Morphology> = emptyList(),
  val posTag: String?
) : RealToken {

  /**
   * @param dependencyRelation the dependency relation of this token
   *
   * @return a new syntactic token
   */
  fun toSyntacticToken(dependencyRelation: DependencyRelation) = Word(
    id = this.id,
    form = this.form,
    position = this.position,
    morphologies = emptyList(), // TODO: set it
    dependencyRelation = dependencyRelation)
}
