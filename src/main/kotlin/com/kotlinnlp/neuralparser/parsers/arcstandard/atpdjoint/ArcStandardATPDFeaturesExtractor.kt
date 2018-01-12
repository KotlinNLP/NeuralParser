/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.templates.featuresextractor.ATWFeaturesExtractor
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.templates.featuresextractor.EmbeddingsFeaturesExtractor
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Root
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.StateView
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.utils.getItemOrNull

/**
 * The [EmbeddingsFeaturesExtractor] that extracts features combining word Embeddings and ambiguous POS vectors for the
 * ArcStandard transition system.
 *
 * @param actionsVectors the encoding vectors of the actions
 * @param actionsVectorsOptimizer the optimizer of the [actionsVectors]
 * @param actionsEncoderOptimizer the optimizer of the RNN used to encode actions
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardATPDFeaturesExtractor(
  actionsVectors: ActionsVectorsMap,
  actionsVectorsOptimizer: ActionsVectorsOptimizer,
  actionsEncoderOptimizer: ParamsOptimizer<NetworkParameters>,
  private val deprelTags: DictionarySet<Deprel>,
  private val posTags: DictionarySet<POSTag>,
  actionsEncodingSize: Int
) : ATWFeaturesExtractor<
  StackBufferState,
  ArcStandardTransition,
  TokensAmbiguousPOSContext>
(
  actionsVectors = actionsVectors,
  actionsVectorsOptimizer = actionsVectorsOptimizer,
  actionsEncoderOptimizer = actionsEncoderOptimizer,
  actionsEncodingSize = actionsEncodingSize
) {

  /**
   * The actions embeddings map key of the transition of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.key: Int
    get() = when (this) {
      is Shift -> 0
      is Root -> 1
      is ArcLeft -> 2
      is ArcRight -> 3
      else -> throw RuntimeException("unknown transition")
    }

  /**
   * The actions embeddings map key of the deprel of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.deprelKey: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardATPDFeaturesExtractor.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The actions embeddings map key of the POS tag of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.posTagKey: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardATPDFeaturesExtractor.posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * Get the tokens window respect to a given state
   *
   * @param stateView a view of the state
   *
   * @return the tokens window as list of Int
   */
  override fun getTokensWindow(stateView: StateView<StackBufferState>) = listOf(
    stateView.state.stack.getItemOrNull(-3),
    stateView.state.stack.getItemOrNull(-2),
    stateView.state.stack.getItemOrNull(-1),
    stateView.state.buffer.getItemOrNull(0))
}
