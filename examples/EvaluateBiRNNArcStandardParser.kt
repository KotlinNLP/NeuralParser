/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.parsers.arcstandard.simple.BiRNNArcStandardParser
import com.kotlinnlp.neuralparser.utils.Timer
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate the model of a [BiRNNArcStandardParser].
 *
 * Command line arguments:
 *  1. The size of the beam
 *  2. The max number of parallel threads
 *  3. The file path of the model
 *  4. The file path of the validation set
 */
fun main(args: Array<String>) {

  val beamSize = args[0].toInt()
  val maxParallelThreads = args[1].toInt()

  println("Loading model from '${args[2]}'.")

  val parser = BiRNNArcStandardParser(
    model = NeuralParserModel.load(FileInputStream(File(args[2]))),
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads)

  val validator = Validator(neuralParser = parser, goldFilePath = args[3])

  println("\nBeam size = $beamSize, MaxParallelThreads = $maxParallelThreads\n")

  val timer = Timer()
  val evaluation = validator.evaluate(printCoNLLScriptEvaluation = true)

  println("\n$evaluation")
  println("\nElapsed time: ${timer.formatElapsedTime()}")

  if (beamSize > 1) {
    (parser.syntaxDecoder as BeamDecoder).close()
  }
}

