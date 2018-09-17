/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.configuration.ArcConfiguration
import com.kotlinnlp.dependencytree.configuration.RootConfiguration
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb
import com.kotlinnlp.linguisticdescription.morphology.properties.Mood
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.MutableMorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.Trace
import com.kotlinnlp.linguisticdescription.syntax.SyntaxType

/**
 *
 */
fun MorphoSyntacticSentence.getGovernor(token: MorphoSyntacticToken): MutableMorphoSyntacticToken? =
  token.syntacticRelation.governor?.let { this.getTokenById(it) }

/**
 *
 */
fun MorphoSyntacticSentence.missingRequiredSubject(token: MorphoSyntacticToken): Boolean =
  this.getSubj(token) == null && !token.isImpersonal && token.isActive

/**
 * TODO: find a better solution to match the deprel
 */
fun MorphoSyntacticSentence.getIObj(token: MorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getDependents(token.id).firstOrNull {
    it.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.IndirectObject.baseAnnotation) }
  }

/**
 * TODO: find a better solution to match the deprel
 */
fun MorphoSyntacticSentence.getObj(token: MorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getDependents(token.id).firstOrNull {
    it.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.Object.baseAnnotation) }
  }

/**
 * TODO: find a better solution to match the deprel
 */
fun MorphoSyntacticSentence.getSubj(token: MorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getDependents(token.id).firstOrNull {
    it.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.Subject.baseAnnotation) }
  }

/**
 *
 */
fun MorphoSyntacticSentence.getTraceSubj(token: MorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getSubj(token)?.takeUnless { it !is Trace }

/**
 *
 */
val MorphoSyntacticToken.isImpersonal: Boolean get() = false  // TODO()

/**
 *
 */
val MorphoSyntacticToken.isActive: Boolean get() = true // TODO()

/**
 *
 */
val MorphoSyntacticToken.isVerbNotAux: Boolean get() = this.isVerb && !this.isAux

/**
 *
 */
val MorphoSyntacticToken.isVerb: Boolean get() = this.mainMorphology is Verb

/**
 *
 */
val MorphoSyntacticToken.isVerbInfinite: Boolean get() = TODO()

/**
 * TODO: find a better and safe solution
 */
val MorphoSyntacticToken.isVerbGerundive: Boolean get() = this.mainMorphology.let {
  it as Verb
  it.mood == Mood.Gerund
}

/**
 * TODO: find a better and safe solution
 */
val MorphoSyntacticToken.isVerbParticiple: Boolean get() = this.mainMorphology.let {
  it as Verb
  it.mood == Mood.Participle
}

/**
 * TODO: find a better and safe solution
 */
val MorphoSyntacticToken.isVerbExplicit: Boolean get() = this.mainMorphology.let {
  it as Verb
  it.mood in listOf(Mood.Indicative, Mood.Subjunctive, Mood.Conditional)
}

/**
 * TODO: find a better solution to match the deprel
 */
val MorphoSyntacticToken.isAux: Boolean get() =
  this.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.AuxTense.annotation) }

/**
 * TODO: find a better solution to match the deprel
 */
val MorphoSyntacticToken.isNeg: Boolean get() =
  this.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.RModNeg.annotation) }

/**
 * TODO: find a better solution to match the deprel
 */
val MorphoSyntacticToken.isCoord2Nd: Boolean get() =
  this.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.Coord2Nd.annotation) }

/**
 * TODO: find a better solution to match the deprel
 */
val MorphoSyntacticToken.isRMod: Boolean get() =
  this.syntacticRelation.dependency.labels.any { it.contains(SyntaxType.RMod.annotation) }

/**
 * TODO: find a better way to get the main token morphology
 */
val MorphoSyntacticToken.mainMorphology: SingleMorphology get() = this.morphologies.first().list.first()

/**
 *
 */
fun MorphoSyntacticSentence.toDependencyTree() = DependencyTree(
  elements = this.tokens.map { it.id },
  dependencies = this.tokens.map {
    if (it.syntacticRelation.governor == null)
      RootConfiguration(id = it.id, deprel = it.syntacticRelation.dependency)
    else
      ArcConfiguration(
        dependent = it.id,
        governor = it.syntacticRelation.governor!!,
        deprel = it.syntacticRelation.dependency)
  })
