/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction

import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The decoding support structure of an actions scorer that contains three [FeedforwardNeuralProcessor]s, the first to
 * score the transitions, the second to score the POS tags and the third to score the deprels.
 * (TPD = Transition + POS tag + Deprel).
 *
 * @property transitionProcessor the neural processor used to score the transitions
 * @property posProcessor the neural processor used to score the POS tags
 * @property deprelProcessor the neural processor used to score the deprels
 * @property outputErrorsInit the default initialization of the output errors
 */
open class TPDSupportStructure(
  transitionProcessor: FeedforwardNeuralProcessor<DenseNDArray>,
  val posProcessor:  FeedforwardNeuralProcessor<DenseNDArray>,
  deprelProcessor: FeedforwardNeuralProcessor<DenseNDArray>,
  outputErrorsInit: OutputErrorsInit
) : TDSupportStructure(
  transitionProcessor = transitionProcessor,
  deprelProcessor = deprelProcessor,
  outputErrorsInit = outputErrorsInit)
