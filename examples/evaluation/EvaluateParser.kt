/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.NeuralParserFactory
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.GenericTransitionBasedParser
import com.kotlinnlp.neuralparser.utils.Timer
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate the model of a generic [NeuralParser].
 *
 * Command line arguments:
 *  1. The path of the model file
 *  2. The path of the validation set file
 *  3. The size of the beam (optional)
 *  4. The max number of parallel threads (optional)
 */
fun main(args: Array<String>) {

  val modelPath: String = args[0]
  val validationSetPath: String = args[1]
  val beamSize: Int = if (args.size > 2) args[2].toInt() else 1
  val maxParallelThreads: Int = if (args.size > 3) args[3].toInt() else 1

  require(beamSize > 0 && maxParallelThreads > 0)

  println("Loading model from '$modelPath'.")

  val parser: NeuralParser<*> = NeuralParserFactory(
    model = NeuralParserModel.load(FileInputStream(File(modelPath))),
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads)

  val validator = Validator(neuralParser = parser, sentences = loadSentences(
    type = "validation",
    filePath = validationSetPath,
    maxSentences = null,
    skipNonProjective = false))

  println("\nBeam size = $beamSize, MaxParallelThreads = $maxParallelThreads\n")

  val timer = Timer()
  val evaluation = validator.evaluate()

  println("\n$evaluation")
  println("\nElapsed time: ${timer.formatElapsedTime()}")

  if (parser is GenericTransitionBasedParser && beamSize > 1) {
    (parser.syntaxDecoder as BeamDecoder).close()
  }
}

