/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser

import com.kotlinnlp.neuralparser.parsers.GenericNeuralParser
import com.kotlinnlp.neuralparser.parsers.arceagerspine.BiRNNArcEagerSpineParser
import com.kotlinnlp.neuralparser.parsers.arceagerspine.BiRNNArcEagerSpineParserModel
import com.kotlinnlp.neuralparser.parsers.archybrid.BiRNNArcHybridParser
import com.kotlinnlp.neuralparser.parsers.archybrid.BiRNNArcHybridParserModel
import com.kotlinnlp.neuralparser.parsers.arcstandard.charbased.CharBasedBiRNNArcStandardParser
import com.kotlinnlp.neuralparser.parsers.arcstandard.charbased.CharBasedBiRNNArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.arcstandard.td.BiRNNTDArcStandardParser
import com.kotlinnlp.neuralparser.parsers.arcstandard.td.BiRNNTDArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.arcstandard.simple.BiRNNArcStandardParser
import com.kotlinnlp.neuralparser.parsers.arcstandard.simple.BiRNNArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.arcstandard.tpd.BiRNNTPDArcStandardParser
import com.kotlinnlp.neuralparser.parsers.arcstandard.tpd.BiRNNTPDArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.arcdistance.BiRNNArcDistanceParser
import com.kotlinnlp.neuralparser.parsers.arcdistance.BiRNNArcDistanceParserModel
import com.kotlinnlp.neuralparser.parsers.arcstandard.tpdjoint.BiRNNTPDJointArcStandardParser
import com.kotlinnlp.neuralparser.parsers.arcstandard.tpdjoint.BiRNNTPDJointArcStandardParserModel

/**
 * The factory of a generic [NeuralParser].
 */
object NeuralParserFactory {

  /**
   * Build a generic [NeuralParser] given a [model].
   *
   * @param model the model of a [NeuralParser]
   *
   * @return a [NeuralParser] with the given [model]
   */
  operator fun invoke(model: NeuralParserModel): GenericNeuralParser = when (model) {
    is BiRNNArcStandardParserModel -> BiRNNArcStandardParser(model)
    is BiRNNTDArcStandardParserModel -> BiRNNTDArcStandardParser(model)
    is BiRNNTPDArcStandardParserModel -> BiRNNTPDArcStandardParser(model)
    is BiRNNTPDJointArcStandardParserModel -> BiRNNTPDJointArcStandardParser(model)
    is CharBasedBiRNNArcStandardParserModel -> CharBasedBiRNNArcStandardParser(model)
    is BiRNNArcHybridParserModel -> BiRNNArcHybridParser(model)
    is BiRNNArcEagerSpineParserModel -> BiRNNArcEagerSpineParser(model)
    is BiRNNArcDistanceParserModel -> BiRNNArcDistanceParser(model)
    else -> throw RuntimeException("Invalid parser model")
  }
}
