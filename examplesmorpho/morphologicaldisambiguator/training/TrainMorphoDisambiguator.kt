/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package morphologicaldisambiguator.training

import com.xenomachina.argparser.mainBody
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.lssencoder.tokensencoder.LSSTokensEncoderModel
import com.kotlinnlp.morphodisambiguator.helpers.dataset.Dataset
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import java.io.File
import java.io.FileInputStream

/**
 * Evaluate the model of an [MorpoDisambiguator].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  println("Loading training dataset from '${parsedArgs.trainingSetPath}'...")

  val trainingDataset: Dataset = Dataset.fromFile(
      filename = parsedArgs.trainingSetPath
  )

  val lssEncoderModel: LSSTokensEncoderModel<ParsingToken, ParsingSentence> = LSSTokensEncoderModel(
      parsedArgs.lssModelPath.let {
        println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
        LHRModel.load(FileInputStream(File(it))).lssModel
      })

  println("\n-- START TRAINING ON %d SENTENCES".format(trainingDataset.examples.size))

}