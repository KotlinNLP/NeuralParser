/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arceagerspine

import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.templates.featuresextractor.TokensWindowFeaturesExtractorTrainable
import com.kotlinnlp.neuralparser.templates.supportstructure.multiprediction.MPSupportStructure
import com.kotlinnlp.neuralparser.utils.features.GroupedDenseFeatures
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.utils.MultiMap
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractor
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineState
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineStateView
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext
import com.kotlinnlp.syntaxdecoder.utils.toTransitions
import com.kotlinnlp.syntaxdecoder.utils.toTransitionsMap

/**
 * The FeaturesExtractor that extracts Embeddings as features for the ArcEagerSpine transition system.
 */
class ArcEagerSpineEmbeddingsFeaturesExtractor
  :
  TokensWindowFeaturesExtractorTrainable<
    ArcEagerSpineState,
    ArcEagerSpineTransition,
    TokensEmbeddingsContext,
    GroupedDenseFeatures,
    ArcEagerSpineStateView,
    MPSupportStructure>() {

  /**
   * The group id of this transition.
   */
  private val Transition<ArcEagerSpineTransition, ArcEagerSpineState>.groupId: Int get() = Utils.getGroupId(this)

  /**
   * Extract features using the given [decodingContext] amd [supportStructure].
   *
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the extracted features
   */
  override fun extract(
    decodingContext: DecodingContext<ArcEagerSpineState, ArcEagerSpineTransition, TokensEmbeddingsContext, DenseItem,
      GroupedDenseFeatures>,
    supportStructure: MPSupportStructure): GroupedDenseFeatures {

    val featuresMap = mutableMapOf<Int, MutableMap<Int, DenseNDArray>>()

    decodingContext.actions
      .filter { it.isAllowed }
      .toTransitions()
      .groupBy { it.groupId }
      .forEach { groupId, transitions ->

        featuresMap[groupId] = mutableMapOf()

        transitions.forEach { transition ->

          featuresMap.getValue(groupId)[transition.id] = this.extractViewFeatures(
            stateView = ArcEagerSpineStateView(state = decodingContext.extendedState.state, transition = transition),
            context = decodingContext.extendedState.context
          )
        }
      }

    return GroupedDenseFeatures(featuresMap = MultiMap(featuresMap))

  }

  /**
   * Backward errors through this [FeaturesExtractor], starting from the errors of the features contained in the given
   * [decodingContext].
   * Errors are required to be already set into the given features.
   *
   * @param decodingContext the decoding context that contains extracted features with their errors
   * @param supportStructure the decoding support structure
   * @param propagateToInput a Boolean indicating whether errors must be propagated to the input items
   */
  override fun backward(
    decodingContext: DecodingContext<ArcEagerSpineState, ArcEagerSpineTransition, TokensEmbeddingsContext, DenseItem,
      GroupedDenseFeatures>,
    supportStructure: MPSupportStructure,
    propagateToInput: Boolean) {

    if (propagateToInput) {

      val transitionsMap: Map<Int, ArcEagerSpineTransition> = decodingContext.actions.toTransitionsMap()

      decodingContext.features.errors.errorsMap.forEach { _, transitionId, errors ->

        val itemsWindow: List<Int?> = this.getTokensWindow(
          stateView = ArcEagerSpineStateView(
            state = decodingContext.extendedState.state,
            transition = transitionsMap.getValue(transitionId)))

        val tokensErrors: Array<DenseNDArray> = errors.splitV(decodingContext.extendedState.context.encodingSize)

        this.accumulateItemsErrors(
          items = decodingContext.extendedState.context.items,
          itemsErrors = itemsWindow.zip(tokensErrors))
      }
    }
  }

  /**
   * Get the tokens window respect to a given state
   *
   * @param stateView a view of the state
   *
   * @return the tokens window as list of Int
   */
  override fun getTokensWindow(stateView: ArcEagerSpineStateView): List<Int?> = when (stateView.transition.type) {

    Transition.Type.SHIFT -> listOf(
      stateView.stack[0],
      stateView.buffer[0]
    )

    Transition.Type.ROOT -> listOf(
      stateView.stack[0]
    )

    Transition.Type.ARC_LEFT, Transition.Type.ARC_RIGHT -> listOf(
      stateView.stack[0],
      stateView.buffer[0]
    )

    else -> throw RuntimeException()
  }

  /**
   * @param tokenId a token id
   * @param context a tokens dense context
   *
   * @return the token representation as dense array
   */
  override fun getTokenEncoding(tokenId: Int?, context: TokensEmbeddingsContext): DenseNDArray
    = context.getTokenEncoding(tokenId)

  /**
   * Beat the occurrence of a new example.
   */
  override fun newExample() = Unit

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() = Unit

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() = Unit

  /**
   * Update the trainable components of this FeaturesExtractor.
   */
  override fun update() = Unit

  /**
   * Accumulate the given [itemsErrors] into the related decoding [items].
   *
   * @param items the decoding items
   * @param itemsErrors a list of Pairs <itemId, itemErrors>
   */
  private fun accumulateItemsErrors(items: List<DenseItem>, itemsErrors: List<Pair<Int?, DenseNDArray>>) {

    itemsErrors.forEach { (itemIndex, errors) ->
      if (itemIndex != null) {
        items[itemIndex].accumulateErrors(errors)
      }
    }
  }
}
