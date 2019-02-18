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
import com.kotlinnlp.morphodisambiguator.*
import com.kotlinnlp.morphodisambiguator.helpers.dataset.Dataset
import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.morphodisambiguator.language.MorphoDictionary
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
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

  val trainingDataset: PreprocessedDataset = PreprocessedDataset.fromDataset(
      dataset = Dataset.fromFile(
          filename = parsedArgs.trainingSetPath
      ))

  val validationDataset: PreprocessedDataset = PreprocessedDataset.fromDataset(
      dataset = Dataset.fromFile(
          filename = parsedArgs.validationSetPath
      ))

  val lssEncoderModel: LSSTokensEncoderModel<ParsingToken, ParsingSentence> = LSSTokensEncoderModel(
      parsedArgs.lssModelPath.let {
        println("Loading the LSSEncoder model from the LHRParser model serialized in '$it'...")
        LHRModel.load(FileInputStream(File(it))).lssModel
      })

  val dictionary = MorphoDictionary(trainingDataset.morphoExamples)

  val disambiguator = MorphoDisambiguator(
      model = MorphoDisambiguatorModel(corpusMorphologies = dictionary,
          lssModel = lssEncoderModel,
          biRNNConfig = BiRNNConfig(connectionType = LayerType.Connection.LSTM,
              hiddenActivation = Tanh(),
              numberOfLayers = 1),
          config = MorphoDisambiguatorConfiguration(
              BiRNNHiddenSize = 100,
              BIRNNOutputSize = 100,
              parallelEncodersHiddenSize = 100,
              parallelEncodersActivationFunction = Tanh())
      ))

  val trainer = MorphoDisambiguatorTrainer(disambiguator = disambiguator,
      batchSize = parsedArgs.batchSize,
      epochs = parsedArgs.epochs,
      validator = MorphoDisambiguatorValidator(
          morphoDisambiguator = disambiguator,
          inputSentences = validationDataset.parsingExamples,
          goldSentences = validationDataset.morphoExamples
      ),
      modelFilename = parsedArgs.modelPath,
      updateMethod = ADAMMethod()
  )


  println("\n-- START TRAINING ON %d SENTENCES".format(trainingDataset.morphoExamples.size))
  trainer.train(trainingSentences = trainingDataset.parsingExamples, goldSentences = trainingDataset.morphoExamples)
}