/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.featuresextractor

import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEncodingContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractorTrainable
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.StateView

/**
 * The FeaturesExtractor that extracts features concatenating the encodings of a tokens window.
 */
abstract class TokensWindowFeaturesExtractorTrainable<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>,
  FeaturesType : Features<*, *>,
  in StateViewType : StateView<StateType>,
  in SupportStructureType : DecodingSupportStructure>
  :
  FeaturesExtractorTrainable<
    StateType,
    TransitionType,
    InputContextType,
    DenseItem,
    FeaturesType,
    SupportStructureType>() {

  /**
   * Get the tokens window respect to a given state
   *
   * @param stateView a view of the state
   *
   * @return the tokens window as list of Int
   */
  abstract protected fun getTokensWindow(stateView: StateViewType): List<Int?>

  /**
   * @param tokenId a token id
   * @param context a tokens dense context
   *
   * @return the token representation as dense array
   */
  abstract protected fun getTokenEncoding(tokenId: Int?, context: InputContextType): DenseNDArray

  /**
   * Extract features from the given [stateView] and [context].
   *
   * @param stateView a view of the state
   * @param context a tokens dense context
   *
   * @return a features array
   */
  protected fun extractViewFeatures(stateView: StateViewType, context: InputContextType): DenseNDArray {

    val itemsWindows: List<Int?> = this.getTokensWindow(stateView)

    val features = Array(
      size = itemsWindows.size,
      init = { i -> this.getTokenEncoding(tokenId = itemsWindows[i], context = context) }
    )

    return concatVectorsV(*features)
  }
}
