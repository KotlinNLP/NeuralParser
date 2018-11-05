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
   * A factory of [MorphoPercolator] by language.
   */
  companion object Factory {

    /**
     * Build a [MorphoPercolator] given the working language.
     *
     * @param language the working language of the percolator
     *
     * @return a new morpho-percolator working on the given language
     */
    operator fun invoke(language: Language): MorphoPercolator = when (language) {
      Language.Italian -> MorphoPercolatorIT()
      else -> MorphoPercolatorSimple()
    }
  }

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
    var curIndex = 1

    while (curIndex < inDepthIds.size) {

      val governorId: Int = inDepthIds[curIndex]
      val dependentsIds: List<Int> = dependencyTree.getDependents(governorId)

      dependentsIds.forEach { dependentId ->

        contextConfigurations = contextConfigurations.flatMap { configuration ->

          val governor: MorphoSynToken.Single = tokens[dependencyTree.getPosition(governorId)]
          val dependent: MorphoSynToken.Single = tokens[dependencyTree.getPosition(dependentId)]
          val contextMorphologies: List<SingleMorphology> = this.getContextMorphologies(
            dependent = dependent,
            dependentContextMorpho = configuration.getValue(dependentId).value,
            governor = governor,
            dependency = dependent.syntacticRelation.dependency)

          contextMorphologies.map {
            val scoredMorpho = ScoredSingleMorphology(value = it, score = governor.morphologies.single().score)
            configuration.copyAndReplace(governorId, scoredMorpho)
          }
        }
      }

      curIndex++
    }

    return contextConfigurations
  }

  /**
   * Perform the percolation of morphological features from a dependent to a governor.
   *
   * @param dependent the dependent token
   * @param dependentContextMorpho the context morphology of the dependent
   * @param governor the governor token
   * @param dependency the dependency between the dependent and the governor
   *
   * @return the list of morphologies that are propagated to the governor
   */
  protected abstract fun getContextMorphologies(dependent: MorphoSynToken.Single,
                                                dependentContextMorpho: SingleMorphology,
                                                governor: MorphoSynToken.Single,
                                                dependency: SyntacticDependency): List<SingleMorphology>

  /**
   * Copy this morphology replacing a given property with a new value.
   *
   * @param replacingProperties the properties to replace as pairs of <name, replacing_value>
   *
   * @return a copy of this morphology with the replaced property
   */
  protected fun SingleMorphology.copy(vararg replacingProperties: Pair<String, MorphologyProperty>): SingleMorphology {

    val properties: MutableMap<String, MorphologyProperty> = this.getProperties().toMutableMap()

    replacingProperties.forEach { (name, value) -> properties.replace(name, value) }

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
