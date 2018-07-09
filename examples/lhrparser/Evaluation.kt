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

  val validator = Validator(neuralParser = LHRParser(model), goldFilePath = parsedArgs.validationSetPath)

  val timer = Timer()
  val evaluation = validator.evaluate(printCoNLLScriptEvaluation = true)

  println("\n$evaluation")
  println("\nElapsed time: ${timer.formatElapsedTime()}")
}
