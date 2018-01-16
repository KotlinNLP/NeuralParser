/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworksPool
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The [TPDJointSupportStructure] of an actions scorer that contains:
 * - a [RecurrentNeuralProcessor] to encode the action features;
 * - an [AttentionNetworksPool] to encode the state;
 * - a [FeedforwardNeuralProcessor] to score the transitions;
 * - a [MultiTaskNetwork] to score POS tags and deprels.
 *
 * @property actionRNNEncoder the recurrent neural processor used to encode the Actions Scorer features
 * @property stateAttentionNetworksPool the pool attention networks used to encode the state
 * @property transitionProcessor the neural processor used to score the transitions
 * @property posDeprelNetwork the multi-task network used to score POS tags and deprels
 * @property outputErrorsInit the default initialization of the output errors
 */
class AttentionDecodingSupportStructure(
  val actionRNNEncoder: RecurrentNeuralProcessor<DenseNDArray>,
  val stateAttentionNetworksPool: AttentionNetworksPool<DenseNDArray>,
  transitionProcessor: FeedforwardNeuralProcessor<DenseNDArray>,
  posDeprelNetwork:  MultiTaskNetwork<DenseNDArray>,
  outputErrorsInit: OutputErrorsInit
) : TPDJointSupportStructure(
  transitionProcessor = transitionProcessor,
  posDeprelNetwork = posDeprelNetwork,
  outputErrorsInit = outputErrorsInit
)
