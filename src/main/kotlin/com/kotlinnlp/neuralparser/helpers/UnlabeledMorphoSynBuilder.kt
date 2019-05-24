/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Undefined
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken

/**
 * A helper class that builds a [MorphoSynSentence] from a [ParsingSentence] and a [DependencyTree].
 *
 * @param parsingSentence a parsing sentence
 * @param dependencyTree the tree that represents the dependencies and the grammatical configuration of the sentence
 */
internal class UnlabeledMorphoSynBuilder(
  private val parsingSentence: ParsingSentence,
  private val dependencyTree: DependencyTree.Unlabeled
) {

  /**
   * @return a new morpho-syntactic sentence built from the given [parsingSentence] and [dependencyTree]
   */
  fun buildSentence(): MorphoSynSentence = MorphoSynSentence(
    id = 0,
    confidence = 0.0,
    tokens = this.parsingSentence.tokens.map {token ->
      this.buildSingleToken(tokenId = token.id, governorId = this.dependencyTree.getHead(token.id))
    },
    position = this.parsingSentence.position
  )

  /**
   * @param tokenId the token id
   * @param governorId the governor id or null if it is the top
   *
   * @return a new single morpho-syntactic token built from the given parsing token
   */
  private fun buildSingleToken(tokenId: Int, governorId: Int?): MorphoSynToken.Single {

    val parsingToken: ParsingToken = this.parsingSentence.getTokenById(tokenId)
    val syntacticRelation = SyntacticRelation(
      governor = governorId,
      attachmentScore = this.dependencyTree.getAttachmentScore(tokenId),
      dependency = Undefined(direction = SyntacticDependency.Direction.NULL))

    return if (parsingToken.position != null)
      Word(
        id = tokenId,
        form = parsingToken.form,
        position = parsingToken.position,
        pos = null,
        morphologies = listOf(),
        contextMorphologies = listOf(), // TODO: set it
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null) // TODO: set it
    else
      WordTrace(
        id = tokenId,
        form = parsingToken.form,
        pos = null,
        morphologies = listOf(),
        contextMorphologies = listOf(), // TODO: set it
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null)
  }
}
