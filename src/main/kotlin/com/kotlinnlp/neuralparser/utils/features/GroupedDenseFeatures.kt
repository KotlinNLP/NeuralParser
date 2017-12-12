/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.features

import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiMap
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features

/**
 * The features extracted from a FeaturesExtractor, used as input of the ActionsScorer.
 * The features are represented by a [MultiMap] of dense array. Their relative errors and relevance are also
 * represented by [MultiMap] of dense arrays.
 *
 * @property featuresMap the grouped dense features representation
 */
data class GroupedDenseFeatures(
  val featuresMap: MultiMap<DenseNDArray>
) : Features<GroupedDenseFeaturesErrors, GroupedDenseFeaturesRelevance>()
