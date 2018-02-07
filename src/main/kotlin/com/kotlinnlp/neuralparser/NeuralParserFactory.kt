/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser

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
   *
   * @return a [NeuralParser] with the given [model]
   */
  operator fun invoke(model: NeuralParserModel,
                      beamSize: Int = 1,
                      maxParallelThreads: Int = 1): NeuralParser<*> = when (model) {

    is TransitionBasedParserModel -> TransitionBasedParserFactory(
      model = model,
      beamSize = beamSize,
      maxParallelThreads = maxParallelThreads)

    else -> throw RuntimeException("Invalid parser model")
  }
}
