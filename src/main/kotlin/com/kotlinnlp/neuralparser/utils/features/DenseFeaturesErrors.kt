/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.features

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors

/**
 * The errors object associated to the [DenseFeatures].
 *
 * @property array the dense features errors representation
 */
data class DenseFeaturesErrors(
  val array: DenseNDArray
): FeaturesErrors
