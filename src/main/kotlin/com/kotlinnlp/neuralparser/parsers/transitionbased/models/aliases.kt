/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models

import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine.ArcEagerSpineActionsScorer
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.ArcStandardEmbeddingsFeaturesExtractor
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.compositeprediction.TDSupportStructure
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.multiprediction.MPSupportStructure
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.singleprediction.SPSupportStructure
import com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items.DenseItem

typealias GenericTransitionBasedParser = TransitionBasedParser<*, *, *, *, *, *, *, *>

typealias SimpleArcEagerSpineActionsScorer = ArcEagerSpineActionsScorer<TokensEmbeddingsContext, DenseItem,
  MPSupportStructure>

typealias ArcStandardSPEmbeddingsFeaturesExtractor = ArcStandardEmbeddingsFeaturesExtractor<SPSupportStructure>
typealias ArcStandardCPEmbeddingsFeaturesExtractor = ArcStandardEmbeddingsFeaturesExtractor<TDSupportStructure>
