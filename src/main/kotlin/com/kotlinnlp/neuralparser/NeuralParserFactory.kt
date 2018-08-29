/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser

import com.kotlinnlp.constraints.Constraint
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParserFactory
import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedParserModel

/**
 * The factory of a generic [NeuralParser].
 */
object NeuralParserFactory {

  /**
   * Build a generic [NeuralParser] given a [model].
   *
   * @param model the model of a [NeuralParser]
   * @param beamSize the size of the beam (ignored if the model is not related to a TransitionBasedParser)
   * @param maxParallelThreads the max number of parallel threads (ignored if the model is not related to a
   *                           TransitionBasedParser or the beamSize is 1)
   * @param constraints a list of linguistic constraints (can be null)
   *
   * @return a [NeuralParser] with the given [model]
   */
  operator fun invoke(model: NeuralParserModel,
                      beamSize: Int = 1,
                      maxParallelThreads: Int = 1,
                      constraints: List<Constraint>? = null): NeuralParser<*> = when (model) {

    is TransitionBasedParserModel -> TransitionBasedParserFactory(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    is LHRModel -> LHRParser(model = model, constraints = constraints)

    else -> throw RuntimeException("Invalid parser model")
  }
}
