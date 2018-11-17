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
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.Word
import com.kotlinnlp.linguisticdescription.sentence.token.WordTrace
import com.kotlinnlp.neuralparser.constraints.Constraint
import java.io.File

/**
 * The morphological configuration of a sentence, that associates a single morphology to each token id.
 */
internal typealias MorphoConfiguration = Map<Int, ScoredSingleMorphology>

/**
 * A map of token ids to the list of related linguistic constraints violated.
 */
internal typealias ViolationsMap = Map<Int, List<Constraint>>

/**
 * A mutable [ViolationsMap].
 */
private typealias MutableViolationsMap = MutableMap<Int, List<Constraint>>

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
 * @return all the single tokens associated to their container ID
 */
internal fun flatTokens(tokens: List<MorphoSynToken>): Map<MorphoSynToken.Single, Int> {

  val flatTokens: MutableMap<MorphoSynToken.Single, Int> = mutableMapOf()
  var index = 0
  val dependencies: Map<Int, Int?> = getDependenciesByComponents(tokens)

  tokens.forEach { token ->
    when (token) {
      is MorphoSynToken.Single -> flatTokens[token.copyPositionIndex(index)] = token.id
      is MorphoSynToken.Composite -> token.components.forEach { c ->
        flatTokens[c.copyPositionIndex(index++)] = token.id
      }
    }
  }

  flatTokens.keys.forEach { token ->
    token.updateSyntacticRelation(token.syntacticRelation.copy(governor = dependencies.getValue(token.id)))
  }

  return flatTokens
}

/**
 * Apply a given morphological configuration to this tokens.
 *
 * @param morphoConfig a morphological configuration of this tokens
 */
internal fun List<MorphoSynToken.Single>.applyMorphologies(morphoConfig: MorphoConfiguration) {

  this.forEach { token -> morphoConfig[token.id]?.let { token.setMorphology(it) } }
}

/**
 * Set a single morphology in this token.
 *
 * @param morphology the morphology to set
 */
internal fun MorphoSynToken.Single.setMorphology(morphology: ScoredSingleMorphology) {

  this.removeAllMorphologies()
  this.addMorphology(morphology)
}

/**
 * Collect all the violations returned for each element of this sequence.
 *
 * @param clearOnValid clear the collected violations the first time the callback returns an empty map (default = true)
 * @param callback a callback called on each element of this sequence, that must return a violations map
 *
 * @return all the violations made during the iteration of this sequence
 */
internal fun <T, I : Sequence<T>> I.collectViolations(clearOnValid: Boolean = true,
                                                      callback: (T) -> ViolationsMap): ViolationsMap {

  val allViolationsMap: MutableViolationsMap = mutableMapOf()
  val collectedViolations: MutableList<ViolationsMap> = mutableListOf()

  run loop@ {
    this.forEach {

      val violations: ViolationsMap = callback(it)

      if (violations.isEmpty() && clearOnValid) {

        collectedViolations.clear()

        return@loop

      } else {
        collectedViolations.add(violations)
      }
    }
  }

  collectedViolations.forEach { allViolationsMap.mergeViolations(it) }

  return allViolationsMap
}

/**
 * Merge this violations map with another.
 *
 * @param other another violations map
 *
 * @return this violations map
 */
private fun MutableViolationsMap.mergeViolations(other: ViolationsMap): MutableViolationsMap {

  other.forEach { tokenIndex, violations ->
    this.merge(tokenIndex, violations) { t, u -> t + u }
  }

  return this
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
