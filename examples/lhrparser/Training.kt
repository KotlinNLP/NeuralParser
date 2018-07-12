/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package lhrparser

import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.utils.loadFromTreeBank
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRTrainer
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils.LossCriterionType
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.AffineMerge
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import lhrparser.utils.TrainingArgs

/**
 * Train the [LHRParser].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = TrainingArgs(args)

  val trainingSentences: List<Sentence> = loadSentences(
    type = "training",
    filePath = parsedArgs.trainingSetPath,
    maxSentences = parsedArgs.maxSentences,
    skipNonProjective = parsedArgs.skipNonProjective)

  val corpus: CorpusDictionary = trainingSentences.let {
    println("Creating corpus dictionary...")
    CorpusDictionary(it)
  }

  val parser = buildParser(
    parsedArgs = parsedArgs,
    tokensEncoderModel = buildTokensEncoderModel(parsedArgs, trainingSentences, corpus),
    corpus = corpus)

  val trainer = buildTrainer(parser = parser, parsedArgs = parsedArgs)

  println("\n-- MODEL")
  println(parser.model)

  println("\n-- START TRAINING ON %d SENTENCES".format(trainingSentences.size))
  println(trainer)

  trainer.train(trainingSentences = trainingSentences)
}

/**
 * Load sentences from a CoNLL file.
 *
 * @param type the string that describes the type of sentences
 * @param filePath the file path
 * @param maxSentences the max number of sentences to load
 * @param skipNonProjective whether to skip non-projective sentences
 *
 * @return the list of loaded sentences
 */
fun loadSentences(type: String, filePath: String, maxSentences: Int?, skipNonProjective: Boolean): ArrayList<Sentence> {

  val sentences = ArrayList<Sentence>()

  println("Loading $type sentences from '%s'%s%s...".format(
    filePath,
    maxSentences?.let { " (max $it)" } ?: "",
    if (skipNonProjective) " skipping non-projective" else ""
  ))

  sentences.loadFromTreeBank(
    filePath = filePath,
    skipNonProjective = skipNonProjective,
    maxSentences = maxSentences)

  return sentences
}

/**
 * Build the LHR Parser.
 *
 * @param parsedArgs the parsed command line arguments
 * @param tokensEncoderModel the tokens-encoder model
 * @param corpus the corpus dictionary
 *
 * @return a new parser
 */
fun buildParser(parsedArgs: TrainingArgs,
                tokensEncoderModel: TokensEncoderModel,
                corpus: CorpusDictionary) = LHRParser(model = LHRModel(
  langCode = parsedArgs.langCode,
  corpusDictionary = corpus,
  tokensEncoderModel = tokensEncoderModel,
  contextBiRNNConfig = BiRNNConfig(
    connectionType = LayerType.Connection.LSTM,
    hiddenActivation = Tanh(),
    numberOfLayers = parsedArgs.numOfContextLayers),
  headsBiRNNConfig = BiRNNConfig(
    connectionType = LayerType.Connection.LSTM,
    hiddenActivation = Tanh()),
  useLabeler = !parsedArgs.noLabeler,
  lossCriterionType = LossCriterionType.Softmax,
  predictPosTags = !parsedArgs.noPosPrediction))

fun getEmbeddingKey(sentence: com.kotlinnlp.linguisticdescription.sentence.Sentence<*>, tokenId: Int): String {

  @Suppress("UNCHECKED_CAST")
  sentence as com.kotlinnlp.linguisticdescription.sentence.Sentence<FormToken>

  return sentence.tokens[tokenId].normalizedForm
}

/**
 * Build a tokens-encoder model.
 *
 * @param parsedArgs the parsed command line arguments
 * @param corpus the corpus dictionary
 *
 * @return a new tokens-encoder model
 */
fun buildTokensEncoderModel(parsedArgs: TrainingArgs,
                            sentences: List<Sentence>,
                            corpus: CorpusDictionary): TokensEncoderModel {

  // TODO: reimplement missing encoders

  val embeddingsMap = EmbeddingsMapByDictionary(
    size = parsedArgs.wordEmbeddingSize,
    dictionary = corpus.words)

  val preEmbeddingsMap = parsedArgs.embeddingsPath!!.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EMBDLoader().load(filename = it)
  }

  return EnsembleTokensEncoderModel(models = listOf(
    EmbeddingsEncoderModel(embeddingsMap = preEmbeddingsMap,
      getEmbeddingKey = ::getEmbeddingKey,
      dropoutCoefficient = parsedArgs.wordDropoutCoefficient),
    EmbeddingsEncoderModel(embeddingsMap = embeddingsMap,
      getEmbeddingKey = ::getEmbeddingKey,
      dropoutCoefficient = parsedArgs.wordDropoutCoefficient)),
    outputMergeConfiguration = AffineMerge(
      outputSize = 100,
      activationFunction = Tanh()))
}

/**
 * Build a trainer for a given [LHRParser].
 *
 * @param parser an LHR parser
 * @param parsedArgs the parsed command line arguments
 *
 * @return a trainer for the given [parser]
 */
fun buildTrainer(parser: LHRParser, parsedArgs: TrainingArgs) = LHRTrainer(
  parser = parser,
  epochs = parsedArgs.epochs,
  batchSize = parsedArgs.batchSize,
  validator = Validator(neuralParser = parser, goldFilePath = parsedArgs.validationSetPath),
  modelFilename = parsedArgs.modelPath,
  lhrErrorsOptions = LHRTrainer.LHRErrorsOptions(
    skipPunctuationErrors = parsedArgs.skipPunctuationErrors),
  updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  verbose = !parsedArgs.quiet)
