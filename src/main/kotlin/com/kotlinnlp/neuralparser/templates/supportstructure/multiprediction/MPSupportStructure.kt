/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.supportstructure.multiprediction

import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionScorer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure

/**
 * The decoding support structure that contains a [MultiPredictionScorer] (MP = Multi Prediction).
 *
 * @property scorer the neural scorer used to score actions
 */
open class MPSupportStructure(val scorer: MultiPredictionScorer<DenseNDArray>) : DecodingSupportStructure
