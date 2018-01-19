/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.attention.tpdjoint

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.templates.featuresextractor.AttentionFeaturesExtractor
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.AttentionSupportStructure
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Root
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState

/**
 * The [AttentionFeaturesExtractor] for the [BiRNNAttentionTPDJointArcStandardParser].
 *
 * @param actionsVectors the encoding vectors of the actions
 * @param actionsVectorsOptimizer the optimizer of the [actionsVectors]
 * @param transformLayerOptimizer the optimizer of the transform layers of the Attention Network
 * @param stateAttentionNetworkOptimizer the optimizer of the attention network used to encode the state
 * @param memoryRNNOptimizer the optimizer of the memory RNN
 * @param featuresLayerOptimizer the optimizer of the features layer
 * @param memoryEncodingSize the size of the memory encoding
 * @param featuresSize the size of the features
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardAttentionTPDJointFeaturesExtractor<in SupportStructureType: AttentionSupportStructure>(
  actionsVectors: ActionsVectorsMap,
  actionsVectorsOptimizer: ActionsVectorsOptimizer,
  transformLayerOptimizer: ParamsOptimizer<FeedforwardLayerParameters>,
  stateAttentionNetworkOptimizer: ParamsOptimizer<AttentionNetworkParameters>,
  memoryRNNOptimizer: ParamsOptimizer<NetworkParameters>,
  featuresLayerOptimizer: ParamsOptimizer<FeedforwardLayerParameters>,
  memoryEncodingSize: Int,
  featuresSize: Int,
  private val deprelTags: DictionarySet<Deprel>,
  private val posTags: DictionarySet<POSTag>
) : AttentionFeaturesExtractor<
  StackBufferState,
  ArcStandardTransition,
  TokensAmbiguousPOSContext,
  SupportStructureType>
(
  actionsVectors = actionsVectors,
  actionsVectorsOptimizer = actionsVectorsOptimizer,
  transformLayerOptimizer = transformLayerOptimizer,
  memoryRNNOptimizer = memoryRNNOptimizer,
  stateAttentionNetworkOptimizer = stateAttentionNetworkOptimizer,
  featuresLayerOptimizer = featuresLayerOptimizer,
  memoryEncodingSize = memoryEncodingSize,
  featuresSize = featuresSize
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
      this is DependencyRelation -> this@ArcStandardAttentionTPDJointFeaturesExtractor.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The actions embeddings map key of the POS tag of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.posTagKey: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardAttentionTPDJointFeaturesExtractor.posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }
}
