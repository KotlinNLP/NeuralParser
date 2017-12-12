/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers

import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.parsers.arceagerspine.ArcEagerSpineActionsScorer
import com.kotlinnlp.neuralparser.parsers.arcstandard.ArcStandardEmbeddingsFeaturesExtractor
import com.kotlinnlp.neuralparser.templates.inputcontexts.*
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.*
import com.kotlinnlp.neuralparser.templates.supportstructure.multiprediction.MPSupportStructure
import com.kotlinnlp.neuralparser.templates.supportstructure.singleprediction.SPSupportStructure
import com.kotlinnlp.neuralparser.utils.items.DenseItem

/**
 * Generic NeuralParser
 */

typealias GenericNeuralParser = NeuralParser<*, *, *, *, *, *, *, *>

/**
 * ActionsScorer
 */

typealias SimpleArcEagerSpineActionsScorer = ArcEagerSpineActionsScorer<TokensEmbeddingsContext, DenseItem,
  MPSupportStructure>

/**
 * FeaturesExtractor
 */

typealias ArcStandardSPEmbeddingsFeaturesExtractor = ArcStandardEmbeddingsFeaturesExtractor<SPSupportStructure>
typealias ArcStandardCPEmbeddingsFeaturesExtractor = ArcStandardEmbeddingsFeaturesExtractor<TDSupportStructure>
