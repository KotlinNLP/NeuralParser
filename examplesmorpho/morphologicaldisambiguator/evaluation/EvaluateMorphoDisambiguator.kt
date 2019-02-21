/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package morphologicaldisambiguator.evaluation

import com.kotlinnlp.morphodisambiguator.MorphoDisambiguator
import com.kotlinnlp.morphodisambiguator.MorphoDisambiguatorModel
import com.kotlinnlp.morphodisambiguator.MorphoDisambiguatorValidator
import com.kotlinnlp.morphodisambiguator.helpers.dataset.Dataset
import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.morphodisambiguator.utils.Statistics
import com.kotlinnlp.utils.Timer
import com.xenomachina.argparser.mainBody
import evaluation.CommandLineArguments
import java.io.File
import java.io.FileInputStream


/**
 * Evaluate the model of an [MorpoDisambiguator].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)
  println("Loading evaluation dataset from '${parsedArgs.validationSetPath}'...")

  val validationDataset: PreprocessedDataset = PreprocessedDataset.fromDataset(
      dataset = Dataset.fromFile(
      filename = parsedArgs.validationSetPath
  ))

  val disambiguator = MorphoDisambiguator(
      model = parsedArgs.modelPath.let {
        println("Loading model from '$it'.")
        MorphoDisambiguatorModel.load(FileInputStream(File(it))) as MorphoDisambiguatorModel
      })

  val validator = MorphoDisambiguatorValidator(
      morphoDisambiguator = disambiguator,
      inputSentences = validationDataset.parsingExamples,
      goldSentences = validationDataset.morphoExamples
      )

  val timer = Timer()
  val evaluation: Statistics = validator.evaluate()

  println("\n$evaluation")
  println("\nElapsed time: ${timer.formatElapsedTime()}")
}