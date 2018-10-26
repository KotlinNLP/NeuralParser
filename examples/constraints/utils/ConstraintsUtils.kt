/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Preposition
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Article
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Pronoun
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import kotlin.reflect.KClass

/**
 * Convert this token into a [MorphoSynToken].
 *
 * @param tree the dependency tree of the sentence
 * @param nextAvailableId the next ID that can be assigned to new tokens
 * @param morphologies the list of possible morphologies of the token
 *
 * @return a new morpho-syntactic token built from this
 */
internal fun CoNLLToken.toMorphoSyntactic(tree: DependencyTree,
                                          nextAvailableId: Int,
                                          morphologies: List<Morphology>): MorphoSynToken {

  val config: GrammaticalConfiguration = tree.getConfiguration(this.id)!!

  return when (config.type) {

    GrammaticalConfiguration.Type.Single -> buildSingleToken(
      id = this.id,
      form = this.form,
      head = if (this.head!! > 0) this.head else null,
      config = config,
      morphologies = morphologies)

    GrammaticalConfiguration.Type.Multiple -> buildCompositeToken(
      id = this.id,
      form = this.form,
      head = if (this.head!! > 0) this.head else null,
      config = config,
      nextAvailableId = nextAvailableId,
      morphologies = morphologies)
  }
}

/**
 * Explode a list of morpho-syntactic tokens into a list of Single tokens, adding all the components of the
 * Composite ones.
 *
 * @param tokens a list of morpho-syntactic tokens
 *
 * @return a new list with all the Single tokens of the given list
 */
internal fun explodeTokens(tokens: List<MorphoSynToken>): List<MorphoSynToken.Single> {

  val explodedTokens: MutableList<MorphoSynToken.Single> = mutableListOf()
  var index = 0
  val dependencies: Map<Int, Int?> = getDependenciesByComponents(tokens)

  tokens.forEach { token ->
    when (token) {
      is MorphoSynToken.Single -> explodedTokens.add(token.copyPositionIndex(index))
      is MorphoSynToken.Composite -> token.components.forEach { c ->
        explodedTokens.add(c.copyPositionIndex(index++))
      }
    }
  }

  explodedTokens.forEach { token ->
    token.updateSyntacticRelation(token.syntacticRelation.copy(governor = dependencies.getValue(token.id)))
  }

  return explodedTokens
}

/**
 * @param id the id
 * @param form the form
 * @param head the head
 * @param config the grammatical configuration
 * @param morphologies the list of possible morphologies of the token
 *
 * @return a new single token
 */
private fun buildSingleToken(id: Int,
                             form: String,
                             head: Int?,
                             config: GrammaticalConfiguration,
                             morphologies: List<Morphology>) = Word(
  id = id,
  form = form,
  position = Position(index = id - 1, start = 0, end = 0),
  pos = config.components.single().pos,
  morphologies = morphologies.map { ScoredSingleMorphology(value = it.components.single(), score = 1.0) },
  contextMorphologies = listOf(),
  syntacticRelation = SyntacticRelation(
    governor = head,
    dependency = config.components.single().syntacticDependency,
    attachmentScore = 1.0
  ),
  coReferences = null,
  semanticRelations = null
)

/**
 * @param id the id
 * @param form the form
 * @param head the head
 * @param config the grammatical configuration
 * @param nextAvailableId the next id that can be assigned to a new token of the sentence (as component)
 * @param morphologies the list of possible morphologies of the token
 *
 * @return a new single token
 */
private fun buildCompositeToken(id: Int,
                                form: String,
                                head: Int?,
                                config: GrammaticalConfiguration,
                                nextAvailableId: Int,
                                morphologies: List<Morphology>): MorphoSynToken.Composite {

  val isPrepArt: Boolean = config.isPrepArt()
  val isVerbEnclitic: Boolean = config.isVerbEnclitic()

  return MorphoSynToken.Composite(
    id = id,
    form = form,
    position = Position(index = id - 1, start = 0, end = 0),
    components = config.components.mapIndexed { i, component ->
      Word(
        id = nextAvailableId + i,
        form = form,
        position = Position(index = id - 1, start = 0, end = 0),
        pos = component.pos,
        morphologies = morphologies.map { ScoredSingleMorphology(value = it.components[i], score = 1.0) },
        contextMorphologies = listOf(),
        syntacticRelation = SyntacticRelation(
          governor = when {
            i == 0 || isPrepArt -> head
            isVerbEnclitic -> nextAvailableId // the ID of the first component
            else -> null
          },
          dependency = component.syntacticDependency,
          attachmentScore = 1.0
        ),
        coReferences = null,
        semanticRelations = null)
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
private fun <T : SingleMorphology> GrammaticalConfiguration.Component.isSubTypeOf(morphologyClass: KClass<T>): Boolean
  = this.pos.let { it is POSTag.Base && it.type.isSubTypeOf(morphologyClass) }

/**
 * Get the dependencies of the given tokens considering the components in case of Composite tokens.
 *
 * @param tokens a list of morpho-syntactic tokens
 *
 * @return the map of Single token ids to their heads
 */
private fun getDependenciesByComponents(tokens: List<MorphoSynToken>): Map<Int, Int?> {

  val dependencies: MutableMap<Int, Int?> = tokens
    .flatMap {
      when (it) {
        is MorphoSynToken.Single -> listOf(it)
        is MorphoSynToken.Composite -> it.components
      }
    }
    .associate { it.id to it.syntacticRelation.governor }
    .toMutableMap()

  tokens.forEach {
    if (it is MorphoSynToken.Composite)
      dependencies
        .filterValues { governorId -> governorId == it.id }
        .keys
        .forEach { dependentId -> dependencies[dependentId] = it.components.first().id }
  }

  return dependencies
}

/**
 * Copy a Single Token assigning a new position index to it.
 *
 * @param index the new position index
 *
 * @return the same token with the position index replaced
 */
private fun MorphoSynToken.Single.copyPositionIndex(index: Int): MorphoSynToken.Single = when (this) {
  is Word -> Word(
    id = this.id,
    form = this.form,
    position = this.position.copy(index = index),
    pos = this.pos,
    morphologies = this.morphologies,
    contextMorphologies = this.contextMorphologies,
    syntacticRelation = this.syntacticRelation,
    coReferences = this.coReferences,
    semanticRelations = this.semanticRelations)
  is WordTrace -> this
  else -> throw RuntimeException("Invalid token type: only Word and WordTrace are allowed at this step.")
}
