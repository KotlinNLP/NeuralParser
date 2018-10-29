/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.POS
import com.kotlinnlp.linguisticdescription.morphology.ScoredMorphology
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.linguisticdescription.syntax.SyntacticType
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.selector.LabelerSelector

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
   * TODO: Set missing properties.
   *
   * @param labelerSelector a labeler prediction selector
   *
   * @return a new morpho-syntactic sentence built from the given [parsingSentence]
   */
  fun buildSentence(labelerSelector: LabelerSelector): MorphoSynSentence = MorphoSynSentence(
    id = 0,
    confidence = 0.0,
    dateTimes = if (this.parsingSentence.dateTimes.isNotEmpty()) this.parsingSentence.dateTimes else null,
    entities = null,
    tokens = this.parsingSentence.tokens.mapIndexed { i, it ->

      // TODO: set the morphologies scores adding the labeler prediction scores of configurations with the same pos
      val morphologies: List<ScoredMorphology> = labelerSelector.getValidMorphologies(
        sentence = this.parsingSentence,
        tokenIndex = i,
        configuration = this.dependencyTree.getConfiguration(it.id)!!
      ).map { ScoredMorphology(components = it.components, score = 1.0) }

      this.buildToken(tokenId = it.id, morphologies = morphologies)
    }
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

    val config: GrammaticalConfiguration = dependencyTree.getConfiguration(tokenId)!!

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

    val newToken = MorphoSynToken.Composite(
      id = parsingToken.id,
      form = parsingToken.form,
      position = checkNotNull(parsingToken.position) { "Composite words must have a position." },
      components = config.components.mapIndexed { i, component ->
        this.buildSingleToken(
          tokenId = tokenId,
          componentId = this.nextAvailableId + i,
          governorId = this.getComponentGovernorId(tokenId = tokenId, componentIndex = i),
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

  /**
   * Get the ID of the governor of a component of a composite token.
   *
   * @param tokenId the ID of a parsing token
   * @param componentIndex the index of a component of the token
   *
   * @return the ID of the governor of the given component
   */
  private fun getComponentGovernorId(tokenId: Int, componentIndex: Int): Int? {

    val governorId: Int? = this.dependencyTree.getHead(tokenId)
    val config: GrammaticalConfiguration = this.dependencyTree.getConfiguration(tokenId)!!

    val isContin: Boolean = config.isContin()
    val isPrepArt: Boolean = config.isPrepArt()
    val isVerbEnclitic: Boolean = config.isVerbEnclitic()

    return when {
      componentIndex == 0 -> governorId
      isPrepArt && !isContin -> governorId
      isPrepArt && isContin -> this.getMultiWordGovernorId(tokenId)
      isVerbEnclitic -> this.nextAvailableId // the ID of the first component
      else -> null
    }
  }

  /**
   * @param tokenId the id of a token that is part of a multi-word
   *
   * @return the governor id of the multi-word of which the given token is part of
   */
  private fun getMultiWordGovernorId(tokenId: Int): Int? {

    var multiWordStartId: Int = this.dependencyTree.getHead(tokenId)!!

    while (this.dependencyTree.getConfiguration(multiWordStartId)!!.isContin())
      multiWordStartId = this.dependencyTree.getHead(tokenId)!!

    return this.dependencyTree.getHead(multiWordStartId)
  }

  /**
   * @return true if this configuration defines the continuation of a multi-word, otherwise false
   */
  private fun GrammaticalConfiguration.isContin(): Boolean =
    this.components.size > 1 && this.components.any { it.syntacticDependency.isSubTypeOf(SyntacticType.Contin) }

  /**
   * @return true if this configuration defines a composite PREP + ART, otherwise false
   */
  private fun GrammaticalConfiguration.isPrepArt(): Boolean =
    this.components.size == 2 &&
      this.components[0].pos.isSubTypeOf(POS.Prep) &&
      this.components[1].pos.isSubTypeOf(POS.Art)

  /**
   * @return true if this configuration defines a composite VERB + PRON, otherwise false
   */
  private fun GrammaticalConfiguration.isVerbEnclitic(): Boolean =
    this.components.size >= 2 &&
      this.components[0].pos.isSubTypeOf(POS.Verb) &&
      this.components.subList(1, this.components.size).all { it.pos.isSubTypeOf(POS.Pron) }

  /**
   * @param syntacticType a syntactic type
   *
   * @return true if the type of this dependency is a subtype of the given syntactic type, otherwise false
   */
  private fun SyntacticDependency?.isSubTypeOf(syntacticType: SyntacticType): Boolean =
    this is SyntacticDependency.Base && this.type.isComposedBy(syntacticType)

  /**
   * @param pos a POS type
   *
   * @return true if the type of this POS tag is a subtype of the given POS, otherwise false
   */
  private fun POSTag?.isSubTypeOf(pos: POS): Boolean = this is POSTag.Base && this.type.isComposedBy(pos)
}
