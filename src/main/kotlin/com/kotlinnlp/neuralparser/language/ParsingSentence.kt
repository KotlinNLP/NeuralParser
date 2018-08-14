/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.properties.datetime.DateTime
import com.kotlinnlp.linguisticdescription.sentence.token.properties.DependencyRelation
import com.kotlinnlp.linguisticdescription.sentence.properties.MultiWords
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.NoFilterSelector

/**
 * The sentence used as input of the [com.kotlinnlp.neuralparser.NeuralParser].
 *
 * @property tokens the list of tokens of the sentence
 * @property multiWords the list of multi-words expressions recognized in the sentence (can be empty)
 * @property dateTimes the list of date-times expressions recognized in the sentence (can be empty)
 */
class ParsingSentence(
  override val tokens: List<ParsingToken>,
  override val multiWords: List<MultiWords> = emptyList(),
  override val dateTimes: List<DateTime> = emptyList()
) : MorphoSentence<ParsingToken> {

  /**
   * Check token ids.
   */
  init {

    require(this.tokens.first().id == 0) { "Tokens ids must start from 0" }

    require(this.tokens.zipWithNext().all { it.second.id == it.first.id + 1 }) {
      "Tokens ids must be incremental and sequential"
    }
  }

  /**
   * A map that associates each token id to each index within the [tokens] list.
   */
  private val tokensIdsToIndices: Map<Int, Int> = this.tokens.withIndex().associate { (i, it) -> it.id to i }

  /**
   * Get the index of a given token within the [tokens] list.
   *
   * @param id the id of a token
   *
   * @throws NoSuchElementException if the given id does not refers to any token of this sentence
   *
   * @return the index of the token within the [tokens] list
   */
  fun getTokenIndex(id: Int): Int = this.tokensIdsToIndices.getValue(id)

  /**
   * TODO: set all properties except for tokens
   *
   * @param dependencyTree the dependency tree from which to extract the dependency relations
   * @param morphoDeprelSelector the morphology and deprel selector
   *
   * @return a new [MorphoSyntacticSentence]
   */
  fun toMorphoSyntacticSentence(dependencyTree: DependencyTree,
                                morphoDeprelSelector: MorphoDeprelSelector = NoFilterSelector()) =
    MorphoSyntacticSentence(
      id = 0,
      confidence = 0.0,
      dateTimes = if (this.dateTimes.isNotEmpty()) this.dateTimes else null,
      entities = null,
      tokens = this.tokens.map {
        it.toMorphoSyntacticToken(
          dependencyRelation = DependencyRelation(
            governor = dependencyTree.heads[it.id],
            deprel = dependencyTree.deprels[it.id]?.label ?: "_",
            attachmentScore = 0.0), // TODO: set it
          morphoDeprelSelector = morphoDeprelSelector
        )
      }
    )
}
