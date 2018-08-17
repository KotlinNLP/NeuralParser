/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package transitionbased

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcrelocate.atpdjoint.BiRNNATPDJointArcRelocateParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcrelocate.atpdjoint.BiRNNATPDJointArcRelocateParserModel
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.ambiguouspos.BiRNNAmbiguousPOSParserTrainer
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.SoftmaxCrossEntropyActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcrelocate.ArcRelocateOracle
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.LogarithmicAccumulator

/**
 * Train a [BiRNNATPDJointArcRelocateParser] with a Softmax activations of the actions scores.
 *
 * Command line arguments:
 *  1. The number of training epochs
 *  2. The file path of the training set
 *  3. The file path of the validation set
 *  4. The file path of the model
 *  5. The file path of the pre-trained word embeddings (optional)
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

  val preTrainedEmbeddings: EmbeddingsMap<String>? = if (args.size > 4) {
    println("Loading pre-trained word embeddings...")
    EmbeddingsMap.load(args[4])
  } else {
    null
  }

  val parserModel = BiRNNATPDJointArcRelocateParserModel(
    actionsScoresActivation = Softmax(),
    scoreAccumulatorFactory = LogarithmicAccumulator.Factory,
    corpusDictionary = corpusDictionary,
    nounDefaultPOSTag = POSTag("NN"),
    otherDefaultPOSTags = listOf(POSTag("JJ")),
    wordEmbeddingSize = 100,
    posEmbeddingSize = 25,
    actionsEmbeddingsSize = 25,
    preTrainedWordEmbeddings = preTrainedEmbeddings,
    biRNNConnectionType = LayerType.Connection.RAN,
    biRNNHiddenActivation = Tanh(),
    biRNNLayers = 2,
    scorerNetworksConfig = ScorerNetworkConfiguration(
      hiddenSize = 100,
      hiddenActivation = Tanh(),
      outputActivation = null),
    appliedActionsNetworkConfig = BiRNNATPDJointArcRelocateParserModel.AppliedActionsNetworkConfiguration(
      outputSize = 100,
      activation = Tanh(),
      connectionType = LayerType.Connection.RAN
    ))

  val parser = BiRNNATPDJointArcRelocateParser(
    model = parserModel,
    wordDropoutCoefficient = 0.25)

  val preprocessor: SentencePreprocessor = BasePreprocessor() // TODO: set with a command line argument

  val trainer = BiRNNAmbiguousPOSParserTrainer(
    neuralParser = parser,
    actionsErrorsSetter = SoftmaxCrossEntropyActionsErrorsSetter(minRelevantError = 1.0e-02),
    oracleFactory = ArcRelocateOracle,
    epochs = epochs,
    batchSize = 1,
    minRelevantErrorsCountToUpdate = 50,
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
