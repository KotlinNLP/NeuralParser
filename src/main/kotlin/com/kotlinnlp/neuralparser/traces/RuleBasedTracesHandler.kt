/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.DependencyRelation
import com.kotlinnlp.linguisticdescription.Deprel
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.Trace
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation

/**
 * A rule-based implementation of the [TracesHandler].
 *
 * @property corefHelpers the list of helpers to handle the co-references
 */
class RuleBasedTracesHandler(private val corefHelpers: List<CorefHelper>) : TracesHandler {

  /**
   * Add the traces to the token list of the given [sentence].
   * For example, add the implicit subjects for null-subject languages.
   *
   * @param sentence the sentence
   */
  override fun addTraces(sentence: MorphoSyntacticSentence) {

    val tree: DependencyTree = sentence.toDependencyTree()
    var lastId: Int = sentence.tokens.maxBy { it.id }!!.id

    sentence.tokens.filter { it.isVerbNotAux }.forEach { token ->

      if (sentence.missingRequiredSubject(token)) {

        sentence.addToken(
          index = this.findBestSubjPosition(
            sentence = sentence,
            tree = tree,
            governor = token),
          token = Trace(
            id = ++lastId,
            morphologies = this.makeTraceMorphology(),
            syntacticRelation = SyntacticRelation(
              governor = token.id,
              dependencyRelation = DependencyRelation(deprel = Deprel(labels = listOf("SUBJ"))), // TODO: set deprel
              attachmentScore = 0.0),
            coReferences = null,
            semanticRelations = null))
      }
    }

    this.corefHelpers.forEach { it.setCoref(sentence) }
  }

  /**
   * @return the morphology of a generic trace
   */
  private fun makeTraceMorphology(): List<ScoredMorphology> =
    listOf(ScoredMorphology(
      type = Morphology.Type.Single,
      list = listOf(SingleMorphology(
        lemma = "", // TODO: it doesn't make sense for the trace
        pos = POS.Pron, // TODO: why don't use a 'Noun' instead?
        allowIncompleteProperties = true)),
      score = 0.0)) // TODO: set it

  /**
   * Find the position of a trace-subject trying to simulate its most common position.
   *
   * For instance, the subject of a verb usually appears before a possible negation or auxiliary
   * (or chains of auxiliaries).
   *
   * @param sentence the sentence
   * @param tree the dependency tree of the sentence
   * @param governor the governor
   *
   * @return the token id after witch to insert the trace of a subject
   */
  private fun findBestSubjPosition(sentence: MorphoSyntacticSentence,
                                   tree: DependencyTree,
                                   governor: MorphoSyntacticToken): Int {

    val token = tree.getLeftDependents(governor.id)
      .map { sentence.getTokenById(it) }
      .firstOrNull { it.isAux || it.isNeg }
      ?: governor

    return sentence.getTokenIndex(token.id)
  }
}
