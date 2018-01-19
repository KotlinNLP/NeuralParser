/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.FeedforwardLayersPool
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [TPDJointSupportStructure] of an actions scorer that contains:
 * - a [RecurrentNeuralProcessor] to encode the action features;
 * - an [AttentionNetworksPool] to encode the state;
 * - a pool of [FeedforwardLayerStructure]s to get the attention arrays;
 * - a pool of [FeedforwardLayerStructure]s to encode the features;
 * - a [FeedforwardNeuralProcessor] to score the transitions;
 * - a [MultiTaskNetwork] to score POS tags and deprels.
 *
 * @property actionMemoryEncoder the action memory RNN
 * @property transformLayersPool the pool of layers used to create the attention arrays of the Attention Network
 * @property stateAttentionNetworksPool the pool attention networks used to encode the state
 * @property transitionProcessor the neural processor used to score the transitions
 * @property featuresLayersPool the pool of Feedforward layers used to encode the features
 * @property posDeprelNetwork the multi-task network used to score POS tags and deprels
 * @property outputErrorsInit the default initialization of the output errors
 */
class AttentionTPDJointSupportStructure(
  override val actionMemoryEncoder: RecurrentNeuralProcessor<DenseNDArray>,
  override val stateAttentionNetworksPool: AttentionNetworksPool<DenseNDArray>,
  override val transformLayersPool: FeedforwardLayersPool<DenseNDArray>,
  override val featuresLayersPool: FeedforwardLayersPool<DenseNDArray>,
  override val transitionProcessor: FeedforwardNeuralProcessor<DenseNDArray>,
  override val posDeprelNetwork:  MultiTaskNetwork<DenseNDArray>,
  override val outputErrorsInit: OutputErrorsInit
) : AttentionSupportStructure, TPDJointSupportStructure {

  /**
   * The last encoded state, as output of the Attention Network.
   */
  override lateinit var lastEncodedState: DenseNDArray
}
