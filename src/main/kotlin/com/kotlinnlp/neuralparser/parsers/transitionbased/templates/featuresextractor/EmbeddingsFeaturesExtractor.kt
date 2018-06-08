/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.featuresextractor

import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensEncodingContext
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.*
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.StateView

/**
 * The FeaturesExtractor that extracts Embeddings as features for the ArcStandard transition system.
 */
abstract class EmbeddingsFeaturesExtractor<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>,
  in SupportStructureType : DecodingSupportStructure>
  :
  TWFeaturesExtractorTrainable<
    StateType,
    TransitionType,
    InputContextType,
    DenseFeatures,
    StateView<StateType>,
    SupportStructureType>() {

  /**
   * Extract features using the given [decodingContext] amd [supportStructure].
   *
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the extracted features
   */
  override fun extract(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: SupportStructureType
  ): DenseFeatures {

    return DenseFeatures(array = this.extractViewFeatures(
      stateView = StateView(state = decodingContext.extendedState.state),
      context = decodingContext.extendedState.context))
  }

  /**
   * @param tokenId a token id
   * @param context a tokens dense context
   *
   * @return the token representation as dense array
   */
  override fun getTokenEncoding(tokenId: Int?, context: InputContextType): DenseNDArray =
    context.getTokenEncoding(tokenId)

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
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: SupportStructureType,
    propagateToInput: Boolean
  ) {

    if (propagateToInput) {

      val itemsWindow: List<Int?> = this.getTokensWindow(StateView(state = decodingContext.extendedState.state))

      val errors: DenseNDArray = decodingContext.features.errors.array
      val tokensErrors: List<DenseNDArray> = errors.splitV(decodingContext.extendedState.context.encodingSize)

      this.accumulateItemsErrors(decodingContext = decodingContext, itemsErrors = itemsWindow.zip(tokensErrors))
    }
  }

  /**
   * Accumulate the given [itemsErrors] into the related items of the given [decodingContext].
   *
   * @param decodingContext the decoding context that contains extracted features with their errors
   * @param itemsErrors a list of Pairs <itemId, itemErrors>
   */
  private fun accumulateItemsErrors(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    itemsErrors: List<Pair<Int?, DenseNDArray>>) {

    itemsErrors.forEach {
      decodingContext.extendedState.context.accumulateItemErrors(itemIndex = it.first, errors = it.second)
    }
  }
}
