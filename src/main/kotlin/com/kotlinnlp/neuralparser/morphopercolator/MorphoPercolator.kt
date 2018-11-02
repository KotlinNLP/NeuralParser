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
import com.kotlinnlp.linguisticdescription.morphology.properties.MorphologyProperty
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Number
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency

/**
 * A helper that performs the percolation of morphological properties from a dependent to its governor.
 */
abstract class MorphoPercolator {

  /**
   * The language for which the percolator works.
   */
  abstract val language: Language

  /**
   * Apply the features percolation to the tokens and return all the possible configurations of context morphologies.
   *
   * @param tokens the list of input tokens
   * @param dependencyTree the dependency tree that defines the dependencies of the tokens
   *
   * @return the list of possible configurations of context morphologies for each token
   */
  fun applyPercolation(tokens: List<MorphoSynToken.Single>,
                       dependencyTree: DependencyTree): List<Map<Int, ScoredSingleMorphology>> {

    var contextConfigurations: List<Map<Int, ScoredSingleMorphology>> =
      listOf(tokens.associate { it.id to it.morphologies.single() }.toMutableMap())

    val inDepthIds: List<Int> = dependencyTree.inDepthPostOrder()
    var curIndex = 0

    while (curIndex < inDepthIds.size) {

      val dependentId: Int = inDepthIds[curIndex]
      val governorId: Int? = dependencyTree.getHead(dependentId)

      if (governorId != null) {

        contextConfigurations = contextConfigurations.flatMap { configuration ->

          val dependent: MorphoSynToken.Single = tokens[dependencyTree.getPosition(dependentId)]
          val governor: MorphoSynToken.Single = tokens[dependencyTree.getPosition(governorId)]
          val governorScoredMorpho: ScoredSingleMorphology = governor.morphologies.single()

          val contextMorphologies: List<SingleMorphology> = this.getContextMorphologies(
            dependentContextMorpho = configuration.getValue(dependentId).value,
            governorMorpho = governorScoredMorpho.value,
            dependency = dependent.syntacticRelation.dependency)

          contextMorphologies.map {
            val scoredMorpho = ScoredSingleMorphology(value = it, score = governorScoredMorpho.score)
            configuration.copyAndReplace(governorId, scoredMorpho)
          }
        }
      }

      curIndex++
    }

    return contextConfigurations
  }

  /**
   * Perform the features percolation from a dependent to a governor given their morphologies and dependency.
   *
   * @param dependentContextMorpho the context morphology of the dependent
   * @param governorMorpho the morphology of the governor
   * @param dependency the dependency between the dependent and the governor
   *
   * @return the list of morphologies that are propagated to the governor
   */
  protected abstract fun getContextMorphologies(dependentContextMorpho: SingleMorphology,
                                                governorMorpho: SingleMorphology,
                                                dependency: SyntacticDependency): List<SingleMorphology>

  /**
   * Copy this morphology replacing a given property with a new value.
   *
   * @param propertyName the name of the property to replace
   * @param property the new property value
   *
   * @return a copy of this morphology with the replaced property
   */
  protected fun SingleMorphology.copy(propertyName: String, property: MorphologyProperty): SingleMorphology {

    val properties: MutableMap<String, MorphologyProperty> = this.getProperties().toMutableMap()

    properties.replace(propertyName, property)

    return SingleMorphology(
      lemma = this.lemma,
      pos = this.pos,
      numericForm = (this as? Number)?.numericForm,
      properties = properties)
  }

  /**
   * @param key a key present in this map
   * @param newValue the new value to associate to the given key
   *
   * @return a new mutable map with all the key-value associations of this map except for the given one
   */
  private fun <K, V>Map<K, V>.copyAndReplace(key: K, newValue: V): Map<K, V> =
    this.toMutableMap().apply { replace(key, newValue) }
}
