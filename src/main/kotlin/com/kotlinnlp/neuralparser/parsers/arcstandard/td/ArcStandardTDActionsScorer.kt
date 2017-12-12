/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.td

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.templates.actionsscorer.TDEmbeddingsActionsScorer
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Root

/**
 * The ArcStandardActionsScorer.
 *
 * @param transitionNetwork a neural network to score the transitions
 * @param deprelNetwork a neural network to score the deprels
 * @param transitionOptimizer the optimizer of the [transitionNetwork] params
 * @param deprelOptimizer the optimizer of the [deprelNetwork] params
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardTDActionsScorer(
  transitionNetwork: NeuralNetwork,
  deprelNetwork: NeuralNetwork,
  transitionOptimizer: ParamsOptimizer<NetworkParameters>,
  deprelOptimizer: ParamsOptimizer<NetworkParameters>,
  deprelTags: DictionarySet<Deprel>
) :
  TDEmbeddingsActionsScorer<
    StackBufferState,
    ArcStandardTransition,
    TokensEmbeddingsContext>
  (
    transitionNetwork = transitionNetwork,
    deprelNetwork = deprelNetwork,
    transitionOptimizer = transitionOptimizer,
    deprelOptimizer = deprelOptimizer,
    deprelTags = deprelTags
  ) {

  /**
   * The [transitionNetwork] outcome index of this transition.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.outcomeIndex: Int get() = when(this) {
    is Shift -> 0
    is Root -> 1
    is ArcLeft -> 2
    is ArcRight -> 3
    else -> throw RuntimeException("unknown transition")
  }

  /**
   * The [deprelNetwork] outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.outcomeIndex: Int get() = when {
    this.transition is Shift -> 0
    this is DependencyRelation -> this@ArcStandardTDActionsScorer.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
    else -> throw RuntimeException("unknown action")
  }
}
