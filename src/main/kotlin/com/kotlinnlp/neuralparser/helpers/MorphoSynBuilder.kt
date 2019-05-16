/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.morphology.ScoredMorphology
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.helpers.labelerselector.LabelerSelector

/**
 * A helper class that builds a [MorphoSynSentence] from a [ParsingSentence] and a [DependencyTree].
 *
 * @param parsingSentence a parsing sentence
 * @param dependencyTree the tree that represents the dependencies and the grammatical configuration of the sentence
 */
class MorphoSynBuilder(
  private val parsingSentence: ParsingSentence,
  private val dependencyTree: DependencyTree
) {

  /**
   * The next id that can be assigned to a new token of the sentence, used in case a new single component has to be
   * created.
   */
  private var nextAvailableId: Int = this.parsingSentence.tokens.asSequence().map { it.id }.max()!! + 1

  /**
   * Build the morpho-syntactic sentence using a [LabelerSelector] to select the valid morphologies.
   *
   * @return a new morpho-syntactic sentence built from the given [parsingSentence]
   */
  fun buildSentence(): MorphoSynSentence = MorphoSynSentence(
    id = 0,
    confidence = 0.0,
    tokens = this.parsingSentence.tokens.mapIndexed { i, token ->

      val attachmentScore: Double = this.dependencyTree.getAttachmentScore(token.id)

      val morphologies: List<ScoredMorphology> = this.parsingSentence.getValidMorphologies(
        tokenIndex = i,
        configuration = this.dependencyTree.getConfiguration(token.id)!!
      ).map { morpho ->
        ScoredMorphology(components = morpho.components, score = attachmentScore)
      }

      this.buildToken(tokenId = token.id, morphologies = morphologies)
    },
    position = this.parsingSentence.position
  )

  /**
   * Build a morpho-syntactic token from a given parsing token.
   *
   * @param tokenId the id of a parsing token of the [parsingSentence]
   * @param morphologies the possible morphologies of the token
   *
   * @return a new morpho-syntactic token build from the given parsing token
   */
  fun buildToken(tokenId: Int, morphologies: List<ScoredMorphology>): MorphoSynToken {

    val config: GrammaticalConfiguration = this.dependencyTree.getConfiguration(tokenId)!!

    require(morphologies.all { it.components.size == config.components.size }) {
      "The given morphologies must have the same number of components of the given grammatical configuration."
    }

    return if (config.components.size == 1)
      this.buildSingleToken(
        tokenId = tokenId,
        governorId = this.dependencyTree.getHead(tokenId),
        grammaticalComponent = config.components.single(),
        morphologies = morphologies.map { it.toSingle() })
    else
      this.buildCompositeToken(tokenId = tokenId, morphologies = morphologies)
  }

  /**
   * @param tokenId the id of the new token
   * @param morphologies the list of possible scored morphologies of the token
   *
   * @return a new composite token
   */
  private fun buildCompositeToken(tokenId: Int, morphologies: List<ScoredMorphology>): MorphoSynToken.Composite {

    val parsingToken: ParsingToken = this.parsingSentence.getTokenById(tokenId)
    val config: GrammaticalConfiguration = this.dependencyTree.getConfiguration(tokenId)!!
    val compositeTokenHandler = CompositeTokenHelper(this.dependencyTree)

    val newToken = MorphoSynToken.Composite(
      id = parsingToken.id,
      form = parsingToken.form,
      position = checkNotNull(parsingToken.position) { "Composite words must have a position." },
      components = config.components.mapIndexed { i, component ->
        this.buildSingleToken(
          tokenId = tokenId,
          componentId = this.nextAvailableId + i,
          governorId = compositeTokenHandler.getComponentGovernorId(
            tokenId = tokenId,
            componentIndex = i,
            prevComponentId = if (i > 0) this.nextAvailableId else null),
          grammaticalComponent = component,
          morphologies = morphologies.map { ScoredSingleMorphology(value = it.components[i], score = it.score) }
        ) as Word
      }
    )

    // Attention: the nextAvailableId must be set after the token has been created in order to calculate the
    // components governors correctly.
    this.nextAvailableId += config.components.size

    return newToken
  }

  /**
   * @param tokenId the id of the original token
   * @param componentId the id of the token in case it is a component (otherwise null)
   * @param governorId the id of the governor (null if it is the top)
   * @param grammaticalComponent the grammatical configuration of the token as single component
   * @param morphologies the list of possible scored morphologies of the token
   *
   * @return a new single token
   */
  private fun buildSingleToken(tokenId: Int,
                               componentId: Int? = null,
                               governorId: Int?,
                               grammaticalComponent: GrammaticalConfiguration.Component,
                               morphologies: List<ScoredSingleMorphology>): MorphoSynToken.Single {

    val parsingToken: ParsingToken = this.parsingSentence.getTokenById(tokenId)
    val syntacticRelation = SyntacticRelation(
      governor = governorId,
      attachmentScore = this.dependencyTree.getAttachmentScore(tokenId),
      dependency = grammaticalComponent.syntacticDependency)

    return if (parsingToken.position != null)
      Word(
        id = componentId ?: tokenId,
        form = parsingToken.form,
        position = parsingToken.position,
        pos = grammaticalComponent.pos,
        morphologies = morphologies,
        contextMorphologies = listOf(), // TODO: set it
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null) // TODO: set it
    else
      WordTrace(
        id = componentId ?: tokenId,
        form = parsingToken.form,
        pos = grammaticalComponent.pos,
        morphologies = morphologies,
        contextMorphologies = listOf(), // TODO: set it
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null)
  }
}
