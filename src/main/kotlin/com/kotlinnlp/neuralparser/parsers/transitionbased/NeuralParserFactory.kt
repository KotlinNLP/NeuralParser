/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased

import com.kotlinnlp.neuralparser.parsers.transitionbased.models.GenericNeuralParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine.BiRNNArcEagerSpineParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine.BiRNNArcEagerSpineParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.archybrid.BiRNNArcHybridParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.archybrid.BiRNNArcHybridParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.charbased.CharBasedBiRNNArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.charbased.CharBasedBiRNNArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.td.BiRNNTDArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.td.BiRNNTDArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.simple.BiRNNArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.simple.BiRNNArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.tpd.BiRNNTPDArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.tpd.BiRNNTPDArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcdistance.BiRNNArcDistanceParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcdistance.BiRNNArcDistanceParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.atpdjoint.BiRNNATPDJointArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.atpdjoint.BiRNNATPDJointArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.tpdjoint.BiRNNTPDJointArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.tpdjoint.BiRNNTPDJointArcStandardParserModel

/**
 * The factory of a generic [NeuralParser].
 */
object NeuralParserFactory {

  /**
   * Build a generic [NeuralParser] given a [model].
   *
   * @param model the model of a [NeuralParser]
   * @param beamSize the max size of the beam (default = 1)
   * @param maxParallelThreads the max number of threads that can run in parallel (default = 1, ignored if beamSize is 1)
   *
   * @return a [NeuralParser] with the given [model]
   */
  operator fun invoke(model: NeuralParserModel,
                      beamSize: Int = 1,
                      maxParallelThreads: Int = 1): GenericNeuralParser = when (model) {

    is BiRNNArcStandardParserModel -> BiRNNArcStandardParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNTDArcStandardParserModel -> BiRNNTDArcStandardParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNTPDArcStandardParserModel -> BiRNNTPDArcStandardParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNTPDJointArcStandardParserModel -> BiRNNTPDJointArcStandardParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNATPDJointArcStandardParserModel -> BiRNNATPDJointArcStandardParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is CharBasedBiRNNArcStandardParserModel -> CharBasedBiRNNArcStandardParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNArcHybridParserModel -> BiRNNArcHybridParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNArcEagerSpineParserModel -> BiRNNArcEagerSpineParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is BiRNNArcDistanceParserModel -> BiRNNArcDistanceParser(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    else -> throw RuntimeException("Invalid parser model")
  }
}
