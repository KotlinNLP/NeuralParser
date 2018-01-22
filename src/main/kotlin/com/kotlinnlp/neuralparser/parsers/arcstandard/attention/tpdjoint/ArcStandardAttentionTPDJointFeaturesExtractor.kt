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
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.templates.featuresextractor.EmbeddingsFeaturesExtractor
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.TPDJointSupportStructure
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder.RecurrentAttentiveNetwork
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
 * The [EmbeddingsFeaturesExtractor] that extracts features combining word Embeddings and ambiguous POS vectors for the
 * ArcStandard transition system.
 */
class ArcStandardAttentionTPDJointFeaturesExtractor(
  featuresSize: Int,
  actionsVectorsMap: ActionsVectorsMap,
  actionsVectorsOptimizer: ActionsVectorsOptimizer,
  recurrentAttentiveNetwork: RecurrentAttentiveNetwork,
  private val deprelTags: DictionarySet<Deprel>,
  private val posTags: DictionarySet<POSTag>
): AttentionFeaturesExtractor<
  StackBufferState,
  ArcStandardTransition,
  TokensAmbiguousPOSContext,
  TPDJointSupportStructure>
(
  featuresSize = featuresSize,
  actionsVectorsMap = actionsVectorsMap,
  actionsVectorsOptimizer = actionsVectorsOptimizer,
  recurrentAttentiveNetwork = recurrentAttentiveNetwork
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
  override val Transition<ArcStandardTransition, StackBufferState>.Action.deprelKey: Int?
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardAttentionTPDJointFeaturesExtractor.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The actions embeddings map key of the POS tag of this action.
   */
  override val Transition<ArcStandardTransition, StackBufferState>.Action.posTagKey: Int?
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardAttentionTPDJointFeaturesExtractor.posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }
}
