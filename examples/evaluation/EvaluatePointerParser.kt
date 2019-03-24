/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.parsers.pointerparser.PointerParser
import com.kotlinnlp.neuralparser.parsers.pointerparser.PointerParserModel
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.utils.Timer
import com.xenomachina.argparser.mainBody
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate the model of an [LHRParser].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val parser: NeuralParser<*> = PointerParser(
    model = parsedArgs.modelPath.let {
      println("Loading model from '$it'.")
      NeuralParserModel.load(FileInputStream(File(it))) as PointerParserModel
    })

  val validator = Validator(
    neuralParser = parser,
    sentences = loadSentences(
      type = "validation",
      filePath = parsedArgs.validationSetPath,
      maxSentences = null,
      skipNonProjective = false),
    sentencePreprocessor = BasePreprocessor())

  val timer = Timer()
  val evaluation = validator.evaluate()

  println("\n$evaluation")
  println("\nElapsed time: ${timer.formatElapsedTime()}")
}

