/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoToken
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * The token of the [ParsingSentence].
 *
 * @param id the token id, an incremental integer starting from 0 within a sentence
 * @param form the form
 * @param position the position of the token in the text
 * @param morphologies the list of possible morphologies of the token
 * @param posTag the best part-of-speech tags associated to the token (can be null) TODO: find a better solution
 */
data class ParsingToken(
  val id: Int,
  override val form: String,
  override val position: Position,
  override val morphologies: List<Morphology>,
  val posTag: String?
) : MorphoToken, RealToken {

  /**
   * @param dependencyRelation the dependency relation of this token
   *
   * @return a new morpho syntactic token
   */
  fun toMorphoSyntacticToken(dependencyRelation: DependencyRelation): MorphoSyntacticToken = Word(
    id = this.id,
    form = this.form,
    position = this.position,
    morphologies = emptyList(), // TODO: set it
    dependencyRelation = dependencyRelation,
    coReferences = null, // TODO: set it
    semanticRelations = null) // TODO: set it
}
