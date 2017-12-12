/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.features

import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiMap
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesRelevance

/**
 * The relevance object associated to the [GroupedDenseFeatures].
 *
 * @property relevanceMap the grouped dense features relevance representation
 */
data class GroupedDenseFeaturesRelevance(
  val relevanceMap: MultiMap<DenseNDArray>
): FeaturesRelevance
