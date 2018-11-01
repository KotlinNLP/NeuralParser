/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import com.kotlinnlp.conllio.CoNLLReader
import com.kotlinnlp.conllio.Sentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.neuralparser.constraints.Constraint
import java.io.File

/**
 * @param filename the path of the JSON file with the constraints
 *
 * @return the list of constraints read
 */
internal fun loadConstraints(filename: String): List<Constraint> {

  println("Loading linguistic constraints from '$filename'...")

  return (Parser().parse(filename) as JsonArray<*>).map { Constraint(it as JsonObject) }
}

/**
 * @param filename the path of the CoNLL dataset
 *
 * @return the list of CoNLL sentences read
 */
internal fun loadSentences(filename: String): List<Sentence> {

  val sentences = mutableListOf<Sentence>()

  println("Loading sentences from '$filename'...")

  CoNLLReader.forEachSentence(File(filename)) {

    require(it.hasAnnotatedHeads()) { "A dataset with annotated heads is required to check linguistic constraints." }
    it.assertValidCoNLLTree()

    sentences.add(it)
  }

  return sentences
}

/**
 * Flat a list of morpho-syntactic tokens into single tokens, flatting all the components of the Composite ones.
 *
 * @param tokens a list of morpho-syntactic tokens
 *
 * @return a list of all the single tokens
 */
internal fun flatTokens(tokens: List<MorphoSynToken>): List<MorphoSynToken.Single> {

  val flatTokens: MutableList<MorphoSynToken.Single> = mutableListOf()
  var index = 0
  val dependencies: Map<Int, Int?> = getDependenciesByComponents(tokens)

  tokens.forEach { token ->
    when (token) {
      is MorphoSynToken.Single -> flatTokens.add(token.copyPositionIndex(index))
      is MorphoSynToken.Composite -> token.components.forEach { c ->
        flatTokens.add(c.copyPositionIndex(index++))
      }
    }
  }

  flatTokens.forEach { token ->
    token.updateSyntacticRelation(token.syntacticRelation.copy(governor = dependencies.getValue(token.id)))
  }

  return flatTokens
}

/**
 * Get the id of the CoNLL token that generated a given exploded token.
 * This is done considering that the [tokens] list has been built replacing the composite tokens with groups of single
 * tokens with new ids. This new ids are not sequential respect the sequence of original CoNLL token ids.
 *
 * @param token a morpho-syntactic token
 * @param tokens the list of tokens that compose the sentence of the [token]
 *
 * @return the id of the CoNLL token that generated the given [token]
 */
internal fun getCoNLLTokenId(token: MorphoSynToken.Single, tokens: List<MorphoSynToken.Single>): Int {

  (0 until tokens.indexOf(token)).reversed().forEach { tokenIndex ->

    val focusToken: MorphoSynToken.Single = tokens[tokenIndex]
    val nextToken: MorphoSynToken.Single = tokens[tokenIndex + 1]

    if (focusToken.id < nextToken.id - 1 && nextToken.id < token.id) return focusToken.id + 1
  }

  return token.id
}

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
