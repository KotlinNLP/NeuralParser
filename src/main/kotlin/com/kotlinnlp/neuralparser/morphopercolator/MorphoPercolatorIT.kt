/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.morphopercolator

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.linguisticdescription.morphology.ScoredSingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Noun
import com.kotlinnlp.linguisticdescription.morphology.properties.*
import com.kotlinnlp.linguisticdescription.morphology.properties.Number
import com.kotlinnlp.linguisticdescription.morphology.properties.interfaces.Conjugable
import com.kotlinnlp.linguisticdescription.morphology.properties.interfaces.Genderable
import com.kotlinnlp.linguisticdescription.morphology.properties.interfaces.Numerable
import com.kotlinnlp.linguisticdescription.morphology.properties.interfaces.PersonDeclinable
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Auxiliary
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Coordinated
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Determiner

/**
 * A helper that performs the percolation of morphological properties from a dependent to its governor for the Italian
 * language.
 */
class MorphoPercolatorIT : MorphoPercolator() {

  companion object {

    /**
     * The map of Person property values to their weight when they have to be compared.
     */
    private val personToWeight: Map<Person, Int> = mapOf(
      Person.First to 3,
      Person.Second to 2,
      Person.Third to 1,
      Person.Undefined to 0
    )
  }

  /**
   * The language for which the percolator works.
   */
  override val language: Language = Language.Italian

  /**
   * Perform the features percolation from a dependent to a governor given their morphologies and dependency.
   *
   * @param dependent the dependent token
   * @param governor the governor token
   * @param dependentContextMorpho the context morphology of the dependent
   * @param governorContextMorpho the context morphology of the governor
   * @param dependencyTree the dependency tree
   *
   * @return the list of morphologies that are propagated to the governor
   */
  override fun getContextMorphologies(dependent: MorphoSynToken.Single,
                                      governor: MorphoSynToken.Single,
                                      dependentContextMorpho: SingleMorphology,
                                      governorContextMorpho: SingleMorphology,
                                      dependencyTree: DependencyTree): List<SingleMorphology> {

    val governorScoredMorpho: ScoredSingleMorphology = governor.morphologies.single()
    var contextMorphologies: List<SingleMorphology> = listOf(governorContextMorpho)
    val dependency: SyntacticDependency = dependent.syntacticRelation.dependency

    when (dependency) {

      is Coordinated -> contextMorphologies = this.getCoordinatedMorphologies(
        dependent = dependent,
        dependentContextMorpho = dependentContextMorpho,
        governorMorpho = governorScoredMorpho.value,
        governorContextMorphologies = contextMorphologies,
        dependencyTree = dependencyTree)

      is Auxiliary -> contextMorphologies = this.getCompositeVerbMorphologies(
        governorContextMorphologies = contextMorphologies,
        dependentContextMorpho = dependentContextMorpho as Conjugable,
        dependentDependency = dependency,
        governorDependency = governor.syntacticRelation.dependency)
    }

    return contextMorphologies
  }

  /**
   * Update the context morphologies of a governor in case of coordination with a dependent of it.
   *
   * @param dependent the dependent token
   * @param dependentContextMorpho the context morphology of the dependent
   * @param governorMorpho the morphology of the governor
   * @param governorContextMorphologies the current context morphologies of the governor
   * @param dependencyTree the dependency tree
   *
   * @return the context morphologies of the governor updated with the given dependency
   */
  private fun getCoordinatedMorphologies(dependent: MorphoSynToken.Single,
                                         dependentContextMorpho: SingleMorphology,
                                         governorMorpho: SingleMorphology,
                                         governorContextMorphologies: List<SingleMorphology>,
                                         dependencyTree: DependencyTree): List<SingleMorphology> {

    var updatedMorphologies: List<SingleMorphology> = governorContextMorphologies

    if (dependentContextMorpho is PersonDeclinable && governorMorpho is PersonDeclinable) {

      val dependentWeight: Int = personToWeight.getValue(dependentContextMorpho.person)
      val governorWeight: Int = personToWeight.getValue(governorMorpho.person)

      if (dependentWeight > governorWeight)
        updatedMorphologies = updatedMorphologies.map { it.copy("person" to dependentContextMorpho.person) }
    }

    if (dependentContextMorpho is Genderable && governorMorpho is Genderable)
      if (dependentContextMorpho.gender != governorMorpho.gender)
        updatedMorphologies = updatedMorphologies.map { it.copy("gender" to Gender.Masculine) }

    if (dependentContextMorpho is Numerable && governorMorpho is Numerable) {

      if (governorMorpho.number != Number.Plural)
        updatedMorphologies = updatedMorphologies.map { it.copy("number" to Number.Plural) }

      if (dependent.isDetermined(dependencyTree))
        updatedMorphologies += updatedMorphologies.map { it.copy("number" to Number.Singular) }
    }

    return updatedMorphologies
  }

  /**
   * Update the context morphologies of a governor in case of a composite verb.
   *
   * @param governorContextMorphologies the current context morphologies of the governor
   * @param dependentContextMorpho the context morphology of the dependent
   * @param dependentDependency the syntactic dependency between the dependent and the governor
   * @param governorDependency the syntactic dependency between the governor and its governor
   *
   * @return the context morphologies of the governor updated with the given dependency
   */
  private fun getCompositeVerbMorphologies(governorContextMorphologies: List<SingleMorphology>,
                                           dependentContextMorpho: Conjugable,
                                           dependentDependency: SyntacticDependency,
                                           governorDependency: SyntacticDependency): List<SingleMorphology> {

    dependentContextMorpho as Numerable
    dependentContextMorpho as PersonDeclinable

    val propagatingProperties: MutableList<Pair<String, MorphologyProperty>> = mutableListOf(
      "mood" to dependentContextMorpho.mood,
      "number" to dependentContextMorpho.number,
      "person" to dependentContextMorpho.person)

    when {
      dependentDependency is Auxiliary.Passive && governorDependency !is Auxiliary ->
        propagatingProperties.add("tense" to dependentContextMorpho.tense)
      dependentContextMorpho.mood == Mood.Indicative && dependentContextMorpho.tense == Tense.Future ->
        propagatingProperties.add("tense" to Tense.Future)
      else ->
        propagatingProperties.add("tense" to Tense.Past)
    }

    return governorContextMorphologies.map { it.copy(propagatingProperties) }
  }

  /**
   * @param dependencyTree the dependency tree in which this token is represented
   *
   * @return true if this token is determined, otherwise false
   */
  private fun MorphoSynToken.Single.isDetermined(dependencyTree: DependencyTree): Boolean =
    this.morphologies.single().value is Noun.Proper ||
      dependencyTree.getDependents(this.id).any { dependentId ->
        dependencyTree.getConfiguration(dependentId)!!.components.single().syntacticDependency is Determiner
      }
}
