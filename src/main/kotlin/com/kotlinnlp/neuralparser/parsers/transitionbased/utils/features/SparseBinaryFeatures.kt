/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.utils.features

import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features

/**
 * The features extracted from a FeaturesExtractor, used as input of the ActionsScorer.
 * The features are represented by a sparse binary array. Their relative errors and relevance are represented by
 * dense array and sparse array respectively .
 *
 * @property array the dense features representation
 */
data class SparseBinaryFeatures(
  val array: SparseBinaryNDArray
) : Features<DenseFeaturesErrors, SparseFeaturesRelevance>()
