/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.FeedforwardLayersPool
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure

/**
 * The decoding support structure used by the AttentionFeaturesExtractor.
 */
interface AttentionSupportStructure : DecodingSupportStructure {

  /**
   * The action memory RNN.
   */
  val actionMemoryEncoder: RecurrentNeuralProcessor<DenseNDArray>

  /**
   * The pool of layers used to create the attention arrays of the Attention Network.
   */
  val stateAttentionNetworksPool: AttentionNetworksPool<DenseNDArray>

  /**
   * The pool attention networks used to encode the state.
   */
  val transformLayersPool: FeedforwardLayersPool<DenseNDArray>

  /**
   * The Feedforward layer used to encode the features.
   */
  val featuresLayer: FeedforwardLayerStructure<DenseNDArray>

  /**
   * The last encoded state, as output of the Attention Network.
   */
  var lastEncodedState: DenseNDArray
}
