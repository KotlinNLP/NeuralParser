/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.simple

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.templates.actionsscorer.SPEmbeddingsActionsScorer
import com.kotlinnlp.neuralparser.templates.supportstructure.singleprediction.SPSupportStructure
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer

/**
 * The ArcStandardActionsScorer.
 *
 * @param network a NeuralNetwork
 * @param optimizer the optimizer of the [network] params
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardActionsScorer<in SupportStructureType : SPSupportStructure>(
  private val network: NeuralNetwork,
  private val optimizer: ParamsOptimizer<NetworkParameters>,
  private val deprelTags: DictionarySet<Deprel>
) :
  SPEmbeddingsActionsScorer<
    StackBufferState,
    ArcStandardTransition,
    TokensEmbeddingsContext,
    SupportStructureType>(network, optimizer, deprelTags)
{

  /**
   * The [network] outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.outcomeIndex: Int get() = when {
    this.transition is Shift -> 0
    this is DependencyRelation -> this@ArcStandardActionsScorer.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
    else -> throw RuntimeException("unknown action")
  }
}
