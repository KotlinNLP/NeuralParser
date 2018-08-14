/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.ScoredMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.*
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector

/**
 * The token of the [ParsingSentence].
 *
 * @property id the token id, an incremental integer starting from 0 within a sentence
 * @property form the form
 * @property morphologies the list of possible morphologies of the token
 * @property posTag the best part-of-speech tags associated to the token (can be null)
 * @param position the position of the token in the text (null if it is a trace)
 */
data class ParsingToken(
  val id: Int,
  override val form: String,
  override val morphologies: List<Morphology>,
  val posTag: String? = null, // TODO: find a better solution
  private val position: Position?
) : MorphoToken, FormToken {

  /**
   * @param dependencyRelation the dependency relation of this token
   * @param morphoDeprelSelector the morphology and deprel selector
   *
   * @return a new morpho syntactic token
   */
  fun toMorphoSyntacticToken(dependencyRelation: DependencyRelation,
                             morphoDeprelSelector: MorphoDeprelSelector): MorphoSyntacticToken {

    val morphologies: List<ScoredMorphology> = morphoDeprelSelector.getValidMorphologies(
      token = this,
      morphologies = this.morphologies,
      deprelLabel = dependencyRelation.deprel
    ).map { ScoredMorphology(type = it.type, list = it.list, score = 0.0) } // TODO: set the score?

    return if (this.position != null)
      Word(
        id = this.id,
        form = this.form,
        position = this.position,
        morphologies = morphologies,
        dependencyRelation = dependencyRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null) // TODO: set it
    else
      WordTrace(
        id = this.id,
        form = this.form,
        morphologies = morphologies,
        dependencyRelation = dependencyRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null)
  }
}
