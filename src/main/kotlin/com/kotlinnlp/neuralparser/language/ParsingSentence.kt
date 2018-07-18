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
    id = 0,  // TODO: set it
    confidence = 0.0, // TODO: set it
    tokens = this.tokens.map {
      it.toSyntacticToken(
        dependencyRelation = DependencyRelation(
          governor = dependencyTree.heads[it.position.index]?.let { index -> this.tokens[index] }?.id,
          deprel = dependencyTree.deprels[it.position.index]?.label ?: "_",
          attachmentScore = 0.0) // TODO: set it
      )
    }
  )
}
