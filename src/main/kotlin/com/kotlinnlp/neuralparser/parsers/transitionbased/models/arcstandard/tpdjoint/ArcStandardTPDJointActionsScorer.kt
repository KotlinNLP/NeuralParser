/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.tpdjoint

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.actionsscorer.TPDJointEmbeddingsActionsScorer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkModel
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkParameters
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Root
import com.kotlinnlp.utils.DictionarySet

/**
 * The ArcStandardTPDJointActionsScorer.
 *
 * @param activationFunction the function used to activate the actions scores
 * @param transitionNetwork a neural network to score the transitions
 * @param posDeprelNetworkModel the model of a multi-task network to score POS tags and deprels
 * @param transitionOptimizer the optimizer of the [transitionNetwork] params
 * @param posDeprelOptimizer the optimizer of the [posDeprelNetworkModel] params
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardTPDJointActionsScorer(
  activationFunction: ActivationFunction?,
  transitionNetwork: NeuralNetwork,
  posDeprelNetworkModel: MultiTaskNetworkModel,
  transitionOptimizer: ParamsOptimizer<NetworkParameters>,
  posDeprelOptimizer: ParamsOptimizer<MultiTaskNetworkParameters>,
  deprelTags: DictionarySet<Deprel>,
  posTags: DictionarySet<POSTag>
) :
  TPDJointEmbeddingsActionsScorer<
    StackBufferState,
    ArcStandardTransition,
    TokensAmbiguousPOSContext>
  (
    activationFunction = activationFunction,
    transitionNetwork = transitionNetwork,
    posDeprelNetworkModel = posDeprelNetworkModel,
    transitionOptimizer = transitionOptimizer,
    posDeprelOptimizer = posDeprelOptimizer,
    deprelTags = deprelTags,
    posTags = posTags
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
   * The deprel network outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.deprelOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardTPDJointActionsScorer.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The pos network outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.posTagOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardTPDJointActionsScorer.posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }
}
