/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package lhrparser

import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.utils.Timer
import com.xenomachina.argparser.mainBody
import lhrparser.helpers.ErrorAnalysisHelper
import lhrparser.utils.EvaluationArgs
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate the model of an [LHRParser].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = EvaluationArgs(args)

  println("Loading model from '${parsedArgs.modelPath}'.")
  val model = LHRModel.load(FileInputStream(File(parsedArgs.modelPath)))

  println("Validation set from '${parsedArgs.validationSetPath}'.")

  println("\n-- MODEL")
  println(model)
  println()

  val parser = LHRParser(model)
  val errorAnalysis = ErrorAnalysisHelper(parser)
  val validator = Validator(neuralParser = parser, goldFilePath = parsedArgs.validationSetPath)

  val timer = Timer()

  val evaluation = validator.evaluate(
    printCoNLLScriptEvaluation = true,
    afterParsing = errorAnalysis::calcStats)

  println("\n$evaluation")
  println("\nElapsed time: ${timer.formatElapsedTime()}")

  println("\n-- Error Analysis")
  errorAnalysis.getStats().sortedByDescending { it.second?.first ?: 0.0 }.forEach { (deprel, stats) ->

    if (stats != null) {
      val (avg, std) = stats
      println("%-12s : %.2f%% [+ %.2f%%, - %.2f%%]".format(
        deprel, 100 * avg, 100 * std.first, 100 * std.second))

    } else {
      println("%-12s : Not found".format(deprel))
    }
  }
}
