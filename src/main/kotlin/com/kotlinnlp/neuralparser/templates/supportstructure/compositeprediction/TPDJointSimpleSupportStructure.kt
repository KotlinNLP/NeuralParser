/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The decoding support structure of an actions scorer that contains a [FeedforwardNeuralProcessor] to score the
 * transitions and a [MultiTaskNetwork] to score POS tags and deprels.
 * (TPD = Transition + POS tag + Deprel).
 *
 * @property transitionProcessor the neural processor used to score the transitions
 * @property posDeprelNetwork the multi-task network used to score POS tags and deprels
 * @property outputErrorsInit the default initialization of the output errors
 */
class TPDJointSimpleSupportStructure(
  override val transitionProcessor: FeedforwardNeuralProcessor<DenseNDArray>,
  override val posDeprelNetwork:  MultiTaskNetwork<DenseNDArray>,
  override val outputErrorsInit: OutputErrorsInit
) : TPDJointSupportStructure
