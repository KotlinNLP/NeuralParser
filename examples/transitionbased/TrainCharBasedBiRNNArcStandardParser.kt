/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package transitionbased

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.charbased.CharBasedBiRNNArcStandardParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcstandard.charbased.CharBasedBiRNNArcStandardParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.charbased.CharBasedBiRNNParserTrainer
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.simplednn.core.functionalities.activations.ReLU
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardOracle
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.AverageAccumulator

/**
 * Train a [CharBasedBiRNNArcStandardParser].
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

  val parserModel = CharBasedBiRNNArcStandardParserModel(
    scoreAccumulatorFactory = AverageAccumulator.Factory,
    corpusDictionary = corpusDictionary,
    charEmbeddingSize = 25,
    charsEncodingSize = 25,
    wordEmbeddingSize = 50,
    hanAttentionSize = 25,
    hanConnectionType = LayerType.Connection.GRU,
    hanHiddenActivation = Tanh(),
    biRNNConnectionType = LayerType.Connection.GRU,
    biRNNHiddenActivation = Tanh(),
    scorerNetworkConfig = ScorerNetworkConfiguration(
      inputDropout = 0.4,
      hiddenSize = 100,
      hiddenActivation = ReLU(),
      hiddenDropout = 0.4,
      outputActivation = Sigmoid()))

  val parser = CharBasedBiRNNArcStandardParser(model = parserModel, wordDropoutCoefficient = 0.25)

  val preprocessor: SentencePreprocessor = BasePreprocessor() // TODO: set with a command line argument

  val trainer = CharBasedBiRNNParserTrainer(
    neuralParser = parser,
    oracleFactory = ArcStandardOracle,
    epochs = epochs,
    batchSize = 1,
    minRelevantErrorsCountToUpdate = 50,
    learningMarginThreshold = 0.1, // 10% of the Sigmoid scale
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
