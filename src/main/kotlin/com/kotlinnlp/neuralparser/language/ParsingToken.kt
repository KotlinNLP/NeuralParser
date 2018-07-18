/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * @param id the token id
 * @param form the form
 * @param position the position of the token in its sentence
 * @param lexicalForms the list of morphological interpretations of the token
 * @param posTag the best part-of-speech tags associated to the token (can be null) TODO: find a better solution
 */
data class ParsingToken(
  val id: Int,
  override val form: String,
  override val position: Position,
  val lexicalForms: List<Morphology> = emptyList(),
  val posTag: String?
) : RealToken {


  /**
   * @param dependencyTree the dependency tree from which to extract the dependency relations
   *
   * @return a new [MorphoSyntacticSentence]
   */
  fun toSyntacticToken(dependencyTree: DependencyTree) = Word(
    id = this.id,
    form = this.form,
    position = this.position,
    morphologies = emptyList(), // TODO
    dependencyRelation = DependencyRelation(
      governor = dependencyTree.heads[this.id], // TODO: fix issue index/id
      deprel = dependencyTree.deprels[this.id]?.label ?: "_", // TODO: fix issue index/id
      attachmentScore = 0.0 // TODO
    ))
}
