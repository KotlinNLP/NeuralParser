/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation

/**
 * @property tokens the tokens
 */
class ParsingSentence(override val tokens: List<ParsingToken>) : Sentence<ParsingToken> {

  /**
   * @param dependencyTree the dependency tree from which to extract the dependency relations
   *
   * @return a new [MorphoSyntacticSentence]
   */
  fun toMorphoSyntacticSentence(dependencyTree: DependencyTree) = MorphoSyntacticSentence(
    id = 0,  // TODO
    confidence = 0.0, // TODO
    tokens = this.tokens.map {
      Word(
        id = it.id,
        form = it.form,
        position = it.position,
        lexicalInterpretations = emptyList(), // TODO
        dependencyRelation = DependencyRelation(
          governor = dependencyTree.heads[it.id], // TODO: fix issue index/id
          deprel = dependencyTree.deprels[it.id]?.label ?: "_", // TODO: fix issue index/id
          attachmentScore = 0.0 // TODO
        ))
    }
  )
}