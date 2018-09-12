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
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Root
import com.kotlinnlp.utils.DictionarySet

/**
 * The ArcStandardTPDActionsScorer.
 *
 * @param activationFunction the function used to activate the actions scores
 * @param transitionOptimizer the optimizer of the transition network params
 * @param posOptimizer the optimizer of the POS network params
 * @param deprelOptimizer the optimizer of the deprel network params
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardTPDActionsScorer(
  activationFunction: ActivationFunction?,
  transitionOptimizer: ParamsOptimizer<NetworkParameters>,
  posOptimizer: ParamsOptimizer<NetworkParameters>,
  deprelOptimizer: ParamsOptimizer<NetworkParameters>,
  private val posTags: DictionarySet<POSTag>,
  private val deprelTags: DictionarySet<Deprel>
) : TPDEmbeddingsActionsScorer<StackBufferState, ArcStandardTransition, TokensAmbiguousPOSContext>(
  activationFunction = activationFunction,
  transitionOptimizer = transitionOptimizer,
  posOptimizer = posOptimizer,
  deprelOptimizer = deprelOptimizer) {

  /**
   * The transition network outcome index of this transition.
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
      this is DependencyRelation -> deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The POS network outcome index of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.posTagOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }
}
