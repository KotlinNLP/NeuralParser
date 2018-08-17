/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package transitionbased

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine.BiRNNArcEagerSpineParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine.BiRNNArcEagerSpineParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple.BiRNNParserTrainer
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.HingeLossActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineOracle
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.AverageAccumulator

/**
 * Train a [BiRNNArcEagerSpineParser].
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

  val trainingSentences: List<CoNLLSentence> = loadSentences(
    type = "training",
    filePath = trainingSetPath,
    maxSentences = null,
    skipNonProjective = false)

  val corpusDictionary: CorpusDictionary = trainingSentences.let {
    println("Creating corpus dictionary...")
    CorpusDictionary(it)
  }

  val parserModel = BiRNNArcEagerSpineParserModel(
    scoreAccumulatorFactory = AverageAccumulator.Factory,
    corpusDictionary = corpusDictionary,
    wordEmbeddingSize = 50,
    posEmbeddingSize = 25,
    biRNNConnectionType = LayerType.Connection.LSTM,
    biRNNHiddenActivation = Tanh(),
    biRNNLayers = 1,
    scorerNetworksConfig = ScorerNetworkConfiguration(
      hiddenSize = 100,
      hiddenActivation = Tanh(),
      outputActivation = null))

  val parser = BiRNNArcEagerSpineParser(
    model = parserModel,
    wordDropoutCoefficient = 0.25,
    posDropoutCoefficient = 0.0)

  val preprocessor: SentencePreprocessor = BasePreprocessor() // TODO: set with a command line argument

  val trainer = BiRNNParserTrainer(
    neuralParser = parser,
    oracleFactory = ArcEagerSpineOracle,
    epochs = epochs,
    batchSize = 1,
    minRelevantErrorsCountToUpdate = 50,
    actionsErrorsSetter = HingeLossActionsErrorsSetter(learningMarginThreshold = 1.0),
    validator = Validator(
      neuralParser = parser,
      sentences = loadSentences(
        type = "validation",
        filePath = validationSetPath,
        maxSentences = null,
        skipNonProjective = false),
      sentencePreprocessor = preprocessor),
    modelFilename = modelFilename,
    sentencePreprocessor = preprocessor)

  println("\n-- START TRAINING ON %d SENTENCES".format(trainingSentences.size))

  trainer.train(trainingSentences)
}
