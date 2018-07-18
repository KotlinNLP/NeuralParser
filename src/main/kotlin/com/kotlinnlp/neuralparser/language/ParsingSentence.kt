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
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * @property tokens the tokens
 */
class ParsingSentence(override val tokens: List<ParsingToken>) : Sentence<ParsingToken> {

  /**
   * Check token ids.
   */
  init {

    require(this.tokens.first().id == 0) { "Tokens ids must start from 0" }

    require((1 until this.tokens.size).all { i -> this.tokens[i].id == this.tokens[i - 1].id + 1 }) {
      "Tokens ids must be incremental and sequential"
    }
  }

  /**
   * @param dependencyTree the dependency tree from which to extract the dependency relations
   *
   * @return a new [MorphoSyntacticSentence]
   */
  fun toMorphoSyntacticSentence(dependencyTree: DependencyTree): MorphoSyntacticSentence {

    var end = -2

    return MorphoSyntacticSentence(
      id = 0,  // TODO: set it
      confidence = 0.0, // TODO: set it
      tokens = this.tokens.mapIndexed { i, it ->

        val start = end + 2 // each couple of consecutive tokens is separated by a spacing char
        end = start + it.form.length - 1

        it.toSyntacticToken(
          dependencyRelation = DependencyRelation(
            governor = dependencyTree.heads[it.id],
            deprel = dependencyTree.deprels[it.id]?.label ?: "_",
            attachmentScore = 0.0), // TODO: set it
          position = Position(index = i, start = start, end = end)
        )
      }
    )
  }
}
