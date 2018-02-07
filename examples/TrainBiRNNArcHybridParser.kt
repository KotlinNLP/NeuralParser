/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple.BiRNNParserTrainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.archybrid.BiRNNArcHybridParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.archybrid.BiRNNArcHybridParserModel
import com.kotlinnlp.neuralparser.utils.loadFromTreeBank
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.HingeLossActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.archybrid.ArcHybridNPOracle
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.AverageAccumulator

/**
 * Train a [BiRNNArcHybridParser].
 *
 * Command line arguments:
 *  1. The number of training epochs
 *  2. The file path of the training set
 *  3. The file path of the validation set
 *  4. The file path of the model
 */
fun main(args: Array<String>) {

  val epochs: Int = args[0].toInt()
  val trainingSetPath: String = args[1]
  val validationSetPath: String = args[2]
  val modelFilename: String = args[3]

  println("Loading training sentences...")
  val trainingSentences = ArrayList<Sentence>()
  trainingSentences.loadFromTreeBank(trainingSetPath)

  println("Creating corpus dictionary...")
  val corpusDictionary = CorpusDictionary(sentences = trainingSentences)

  val parserModel = BiRNNArcHybridParserModel(
    scoreAccumulatorFactory = AverageAccumulator.Factory,
    corpusDictionary = corpusDictionary,
    wordEmbeddingSize = 50,
    posEmbeddingSize = 25,
    biRNNConnectionType = LayerType.Connection.LSTM,
    biRNNHiddenActivation = Tanh(),
    biRNNLayers = 1,
    scorerNetworkConfig = ScorerNetworkConfiguration(
      inputDropout = 0.4,
      hiddenSize = 100,
      hiddenActivation = ReLU(),
      hiddenDropout = 0.4,
      outputActivation = Sigmoid()))

  val parser = BiRNNArcHybridParser(
    model = parserModel,
    wordDropoutCoefficient = 0.25,
    posDropoutCoefficient = 0.15)

  val trainer = BiRNNParserTrainer(
    neuralParser = parser,
    oracleFactory = ArcHybridNPOracle,
    epochs = epochs,
    batchSize = 1,
    minRelevantErrorsCountToUpdate = 50,
    actionsErrorsSetter = HingeLossActionsErrorsSetter(learningMarginThreshold = 0.1), // 10% of the Sigmoid scale
    validator = Validator(
      neuralParser = parser,
      goldFilePath = validationSetPath),
    modelFilename = modelFilename)

  println("\n-- START TRAINING ON %d SENTENCES".format(trainingSentences.size))

  trainer.train(trainingSentences)
}
