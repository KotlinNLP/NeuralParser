/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.utils.features

import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesRelevance

/**
 * The relevance object associated to the [SparseBinaryFeatures].
 *
 * @property array the sparse features relevance representation
 */
data class SparseFeaturesRelevance(val array: SparseNDArray): FeaturesRelevance
