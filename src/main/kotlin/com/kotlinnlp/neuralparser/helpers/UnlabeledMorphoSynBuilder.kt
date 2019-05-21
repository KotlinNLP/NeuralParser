/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.helpers.labelerselector.LabelerSelector

/**
 * A helper class that builds a [MorphoSynSentence] from a [ParsingSentence] and a [DependencyTree].
 *
 * @param parsingSentence a parsing sentence
 * @param dependencyTree the tree that represents the dependencies of the sentence
 */
class UnlabeledMorphoSynBuilder(
  private val parsingSentence: ParsingSentence,
  private val dependencyTree: DependencyTree
) {

  /**
   * Build the morpho-syntactic sentence without considering labeling information.
   *
   * @return a new morpho-syntactic sentence built from the given [parsingSentence]
   */
  fun buildSentence(): MorphoSynSentence = MorphoSynSentence(
    id = 0,
    confidence = 0.0,
    tokens = this.parsingSentence.tokens.map { token ->
      this.buildToken(tokenId = token.id,
        governorId = this.dependencyTree.getHead(token.id),
        grammaticalComponent = null) },
    position = this.parsingSentence.position
  )

  /**
   * @param tokenId the id of the original token
   * @param componentId the id of the token in case it is a component (otherwise null)
   * @param governorId the id of the governor (null if it is the top)
   * @param grammaticalComponent the grammatical configuration of the token as single component
   *
   * @return a new single token
   */
  private fun buildToken(tokenId: Int,
                               componentId: Int? = null,
                               governorId: Int?,
                               grammaticalComponent: GrammaticalConfiguration.Component?): MorphoSynToken.Single {

    val parsingToken: ParsingToken = this.parsingSentence.getTokenById(tokenId)
    val syntacticRelation = SyntacticRelation(
      governor = governorId,
      attachmentScore = this.dependencyTree.getAttachmentScore(tokenId),
      dependency = grammaticalComponent?.syntacticDependency ?: SyntacticDependency.invoke(annotation = "_"))

    return if (parsingToken.position != null)
      Word(
        id = componentId ?: tokenId,
        form = parsingToken.form,
        position = parsingToken.position,
        pos = null,
        morphologies = listOf(),
        contextMorphologies = listOf(),
        syntacticRelation = syntacticRelation,
        coReferences = null,
        semanticRelations = null)
    else
      WordTrace(
        id = componentId ?: tokenId,
        form = parsingToken.form,
        pos = null,
        morphologies = listOf(),
        contextMorphologies = listOf(),
        syntacticRelation = syntacticRelation,
        coReferences = null,
        semanticRelations = null)
  }
}
