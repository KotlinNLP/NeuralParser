/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.tpd

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.actionsscorer.TPDEmbeddingsActionsScorer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
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
 * The ArcStandardTPDActionsScorer.
 *
 * @param activationFunction the function used to activate the actions scores
 * @param transitionNetwork a neural network to score the transitions
 * @param posNetwork a neural network to score the posTags
 * @param deprelNetwork a neural network to score the deprels
 * @param transitionOptimizer the optimizer of the [transitionNetwork] params
 * @param posOptimizer the optimizer of the [posNetwork] params
 * @param deprelOptimizer the optimizer of the [deprelNetwork] params
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardTPDActionsScorer(
  activationFunction: ActivationFunction?,
  transitionNetwork: NeuralNetwork,
  posNetwork: NeuralNetwork,
  deprelNetwork: NeuralNetwork,
  transitionOptimizer: ParamsOptimizer<NetworkParameters>,
  posOptimizer: ParamsOptimizer<NetworkParameters>,
  deprelOptimizer: ParamsOptimizer<NetworkParameters>,
  posTags: DictionarySet<POSTag>,
  deprelTags: DictionarySet<Deprel>
) :
  TPDEmbeddingsActionsScorer<
    StackBufferState,
    ArcStandardTransition,
    TokensAmbiguousPOSContext>
  (
    activationFunction = activationFunction,
    transitionNetwork = transitionNetwork,
    posNetwork = posNetwork,
    deprelNetwork = deprelNetwork,
    transitionOptimizer = transitionOptimizer,
    posOptimizer = posOptimizer,
    deprelOptimizer = deprelOptimizer,
    posTags = posTags,
    deprelTags = deprelTags
  ) {

  /**
   * The [transitionNetwork] outcome index of this transition.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.outcomeIndex: Int
    get() = when (this) {
      is Shift -> 0
      is Root -> 1
      is ArcLeft -> 2
      is ArcRight -> 3
      else -> throw RuntimeException("unknown transition")
    }

  /**
   * The [deprelNetwork] outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.deprelOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardTPDActionsScorer.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The [posNetwork] outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.posTagOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardTPDActionsScorer.posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }
}
