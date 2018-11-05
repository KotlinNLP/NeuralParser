/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.morphopercolator

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency

/**
 * A helper that performs the percolation of morphological properties from a dependent to its governor simply copying
 * the context morphology of the dependent.
 */
class MorphoPercolatorSimple : MorphoPercolator() {

  /**
   * The language for which the percolator works.
   */
  override val language: Language = Language.Unknown

  /**
   * Perform the features percolation returning the context morphology of the dependent.
   *
   * @param dependent the dependent token
   * @param dependentContextMorpho the context morphology of the dependent
   * @param governor the governor token
   * @param dependency the dependency between the dependent and the governor
   *
   * @return the list of morphologies that are propagated to the governor
   */
  override fun getContextMorphologies(dependent: MorphoSynToken.Single,
                                      dependentContextMorpho: SingleMorphology,
                                      governor: MorphoSynToken.Single,
                                      dependency: SyntacticDependency): List<SingleMorphology> =
    listOf(dependentContextMorpho)
}
