/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.morphopercolator

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken

/**
 * A helper that performs the percolation of morphological properties from a dependent to its governor simply copying
 * the context morphologies of the governor.
 */
class MorphoPercolatorSimple : MorphoPercolator() {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The language for which the percolator works.
   */
  override val language: Language = Language.Unknown

  /**
   * Perform the features percolation returning the context morphologies of the governor.
   *
   * @param dependent the dependent token
   * @param governor the governor token
   * @param dependentContextMorpho the context morphology of the dependent
   * @param governorContextMorpho the context morphology of the governor
   * @param dependencyTree the dependency tree
   *
   * @return the same list of context morphologies of the governor given as input
   */
  override fun getContextMorphologies(dependent: MorphoSynToken.Single,
                                      governor: MorphoSynToken.Single,
                                      dependentContextMorpho: SingleMorphology,
                                      governorContextMorpho: SingleMorphology,
                                      dependencyTree: DependencyTree): List<SingleMorphology> =
    listOf(governorContextMorpho)
}
