/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package evaluation

import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.parsers.distance.DistanceParser
import com.kotlinnlp.neuralparser.parsers.distance.DistanceParserModel
import com.kotlinnlp.neuralparser.parsers.distance.decoder.DependencyDecoder
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.utils.Timer
import java.io.File
import java.io.FileInputStream
import kotlin.reflect.KClass

/**
 * Evaluate the model of a [DistanceParserModel].
 *
 * @param args the command line arguments
 * @param dependencyDecoderClass the class of the dependency decoder to use to build the parser
 */
internal fun <T : DependencyDecoder>evaluateDistanceParser(args: Array<String>, dependencyDecoderClass: KClass<T>) {

  val parsedArgs = CommandLineArguments(args)

  val parser = DistanceParser(
    model = parsedArgs.modelPath.let {
      println("Loading model from '$it'.")
      NeuralParserModel.load(FileInputStream(File(it))) as DistanceParserModel
    },
    decoderClass = dependencyDecoderClass)

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

