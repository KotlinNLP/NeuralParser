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
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb
import com.kotlinnlp.linguisticdescription.morphology.properties.Mood
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MutableMorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.Trace
import com.kotlinnlp.linguisticdescription.syntax.dependencies.*

/**
 *
 */
fun MorphoSyntacticSentence.getGovernor(token: MutableMorphoSyntacticToken): MutableMorphoSyntacticToken? =
  token.syntacticRelation.governor?.let { this.getTokenById(it) }

/**
 *
 */
fun MorphoSyntacticSentence.missingRequiredSubject(token: MutableMorphoSyntacticToken): Boolean =
  this.getSubj(token) == null && !token.isImpersonal && token.isActive

/**
 *
 */
fun MorphoSyntacticSentence.getIObj(token: MutableMorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getDependents(token.id).firstOrNull { it.isDependentAs(IndirectObject::class) }

/**
 *
 */
fun MorphoSyntacticSentence.getObj(token: MutableMorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getDependents(token.id).firstOrNull { it.isDependentAs(Object::class) }

/**
 *
 */
fun MorphoSyntacticSentence.getSubj(token: MutableMorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getDependents(token.id).firstOrNull { it.isDependentAs(Subject::class) }

/**
 *
 */
fun MorphoSyntacticSentence.getTraceSubj(token: MutableMorphoSyntacticToken): MutableMorphoSyntacticToken? =
  this.getSubj(token)?.takeUnless { it !is Trace }

/**
 *
 */
val MutableMorphoSyntacticToken.isImpersonal: Boolean get() = false  // TODO()

/**
 *
 */
val MutableMorphoSyntacticToken.isActive: Boolean get() = true // TODO()

/**
 *
 */
val MutableMorphoSyntacticToken.isVerbNotAux: Boolean get() = this.isVerb && !this.isAux

/**
 *
 */
val MutableMorphoSyntacticToken.isVerb: Boolean get() = this.flatMorphologies.any { it is Verb }

/**
 *
 */
val MutableMorphoSyntacticToken.isVerbInfinite: Boolean get() = TODO()

/**
 * TODO: find a better and safe solution
 */
val MutableMorphoSyntacticToken.isVerbGerundive: Boolean get() = this.flatMorphologies.any {
  it is Verb && it.mood == Mood.Gerund
}

/**
 * TODO: find a better and safe solution
 */
val MutableMorphoSyntacticToken.isVerbParticiple: Boolean get() = this.flatMorphologies.any {
  it is Verb && it.mood == Mood.Participle
}

/**
 * TODO: find a better and safe solution
 */
val MutableMorphoSyntacticToken.isVerbExplicit: Boolean get() = this.flatMorphologies.any {
  it is Verb && it.mood in listOf(Mood.Indicative, Mood.Subjunctive, Mood.Conditional)
}

/**
 *
 */
val MutableMorphoSyntacticToken.isAux: Boolean get() = this.isDependentAs(Auxiliary.Tense::class)

/**
 *
 */
val MutableMorphoSyntacticToken.isNeg: Boolean get() = this.isDependentAs(RestrictiveModifier.Negative::class)

/**
 *
 */
val MutableMorphoSyntacticToken.isCoord2Nd: Boolean get() = TODO("Coord2Nd class is missing") // this.matches(Coord2Nd::class)

/**
 *
 */
val MutableMorphoSyntacticToken.isRMod: Boolean get() = this.isDependentAs(RestrictiveModifier::class)

/**
 *
 */
fun MorphoSyntacticSentence.toDependencyTree() = DependencyTree(
  elements = this.tokens.map { it.id },
  dependencies = this.tokens.map {
    if (it.syntacticRelation.governor == null)
      RootConfiguration(id = it.id, dependency = it.syntacticRelation.dependency)
    else
      ArcConfiguration(
        dependent = it.id,
        governor = it.syntacticRelation.governor!!,
        dependency = it.syntacticRelation.dependency)
  })
