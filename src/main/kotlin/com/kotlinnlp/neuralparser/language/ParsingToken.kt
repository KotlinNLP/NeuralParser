/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Preposition
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Article
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Pronoun
import com.kotlinnlp.linguisticdescription.sentence.token.*
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.selector.LabelerSelector
import kotlin.reflect.KClass

/**
 * The token of the [ParsingSentence].
 *
 * @property id the id of the token, unique within its sentence
 * @property form the form
 * @property morphologies the list of possible morphologies of the token
 * @property pos the list of part-of-speech tags associated to the token (more for composite tokens, can be null)
 * @param position the position of the token in the text (null if it is a trace)
 */
data class ParsingToken(
  override val id: Int,
  override val form: String,
  override val morphologies: List<Morphology>,
  override val pos: List<POSTag>? = null, // TODO: find a better solution
  private val position: Position?
) : MorphoToken, FormToken, TokenIdentificable {

  /**
   * @param sentence the sentence this token is part of
   * @param tokenIndex the index of this token within the sentence tokens list
   * @param nextIdAvailable the next id that can be assigned to a new token to add to the sentence (as component)
   * @param governorId the governor id
   * @param attachmentScore the attachment score
   * @param config the grammatical configuration of this token
   * @param labelerSelector a labeler prediction selector
   *
   * @return a new morpho syntactic token
   */
  fun toMorphoSynToken(sentence: ParsingSentence,
                       tokenIndex: Int,
                       nextIdAvailable: Int,
                       governorId: Int?,
                       attachmentScore: Double,
                       config: GrammaticalConfiguration,
                       labelerSelector: LabelerSelector): MorphoSynToken {

    // TODO: set the score adding the labeler prediction scores of configurations with the same pos
    val morphologies: List<ScoredMorphology> = labelerSelector.getValidMorphologies(
      sentence = sentence,
      tokenIndex = tokenIndex,
      configuration = config
    ).map { ScoredMorphology(components = it.components, score = 0.0) }

    require(morphologies.all { it.components.size == config.components.size }) {
      "The selected morphologies must have the same number of components of the given grammatical configuration."
    }

    return if (config.components.size == 1)
      this.buildSingleToken(
        id = this.id,
        governorId = governorId,
        attachmentScore = attachmentScore,
        grammaticalComponent = config.components.single(),
        morphologies = morphologies.map { it.toSingle() })
    else
      this.buildCompositeToken(
        nextIdAvailable = nextIdAvailable,
        governorId = governorId,
        attachmentScore = attachmentScore,
        config = config,
        morphologies = morphologies)
  }

  /**
   * @param id the id of the new token
   * @param governorId the governor id
   * @param attachmentScore the attachment score
   * @param grammaticalComponent the grammatical configuration of the token as single component
   * @param morphologies the list of possible scored morphologies of the token
   *
   * @return a new single token
   */
  private fun buildSingleToken(id: Int,
                               governorId: Int?,
                               attachmentScore: Double,
                               grammaticalComponent: GrammaticalConfiguration.Component,
                               morphologies: List<ScoredSingleMorphology>): MorphoSynToken.Single {

    val syntacticRelation = SyntacticRelation(
      governor = governorId,
      attachmentScore = attachmentScore,
      dependency = grammaticalComponent.syntacticDependency)

    return if (this.position != null)
      Word(
        id = id,
        form = this.form,
        position = this.position,
        pos = grammaticalComponent.pos,
        morphologies = morphologies,
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null) // TODO: set it
    else
      WordTrace(
        id = id,
        form = this.form,
        pos = grammaticalComponent.pos,
        morphologies = morphologies,
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null)
  }

  /**
   * @param nextIdAvailable the next id that can be assigned to a new token to add to the sentence (as component)
   * @param governorId the governor id
   * @param attachmentScore the attachment score
   * @param config the grammatical configuration of the token
   * @param morphologies the list of possible scored morphologies of the token
   *
   * @return a new composite token
   */
  private fun buildCompositeToken(nextIdAvailable: Int,
                                  governorId: Int?,
                                  attachmentScore: Double,
                                  config: GrammaticalConfiguration,
                                  morphologies: List<ScoredMorphology>): MorphoSynToken.Composite {

    val isPrepArt: Boolean = config.isPrepArt()
    val isVerbEnclitic: Boolean = config.isVerbEnclitic()

    return MorphoSynToken.Composite(
      id = this.id,
      form = this.form,
      position = checkNotNull(this.position) { "Composite words must have a position." },
      components = config.components.mapIndexed { i, component ->
        this.buildSingleToken(
          id = nextIdAvailable + i,
          governorId = when {
            i == 0 || isPrepArt -> governorId
            isVerbEnclitic -> nextIdAvailable // the ID of the first component
            else -> null
          },
          attachmentScore = attachmentScore,
          grammaticalComponent = component,
          morphologies = morphologies.map { ScoredSingleMorphology(value = it.components[i], score = it.score) }
        ) as Word
      }
    )
  }

  /**
   * @return true if this configuration defines a composite PREP + ART, otherwise false
   */
  private fun GrammaticalConfiguration.isPrepArt(): Boolean =
    this.components.size == 2 &&
      this.components[0].isSubTypeOf(Preposition::class) &&
      this.components[1].isSubTypeOf(Article::class)

  /**
   * @return true if this configuration defines a composite VERB + PRON, otherwise false
   */
  private fun GrammaticalConfiguration.isVerbEnclitic(): Boolean =
    this.components.size >= 2 &&
      this.components[0].isSubTypeOf(Verb::class) &&
      this.components.subList(1, this.components.size).all { it.isSubTypeOf(Pronoun::class) }

  /**
   * @param morphologyClass the KClass of a single morphology
   *
   * @return true if this component defines a POS that is a subtype of the type of given morphology, otherwise false
   */
  private fun <T : SingleMorphology>GrammaticalConfiguration.Component.isSubTypeOf(morphologyClass: KClass<T>): Boolean
    = this.pos.let { it is POSTag.Base && it.type.isSubTypeOf(morphologyClass) }
}
