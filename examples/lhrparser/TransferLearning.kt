/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package lhrparser

import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRTransferLearning
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.POSEmbeddingsKey
import com.kotlinnlp.tokensencoder.embeddings.WordEmbeddingsKey
import com.kotlinnlp.tokensencoder.embeddings.dictionary.EmbeddingsEncoderByDictionaryModel
import com.kotlinnlp.tokensencoder.ensamble.concat.ConcatTokensEncoderModel
import com.kotlinnlp.utils.DictionarySet
import com.xenomachina.argparser.mainBody
import lhrparser.utils.TransferLearningArgs
import java.io.File
import java.io.FileInputStream

/**
 * Train the [LHRParser].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = TransferLearningArgs(args)

  val trainingSentences = loadSentences(
    type = "training",
    filePath = parsedArgs.trainingSetPath,
    maxSentences = parsedArgs.maxSentences,
    skipNonProjective = parsedArgs.skipNonProjective)

  println("Load reference parser with model: ${parsedArgs.referenceModelPath}...")
  val referenceParser = LHRParser(model = LHRModel.load(FileInputStream(File(parsedArgs.referenceModelPath))))

  val targetParser = buildTargetParser(
    parsedArgs = parsedArgs,
    referenceModel = referenceParser.model,
    corpus = trainingSentences.let {
      println("Creating corpus dictionary...")
      CorpusDictionary(it)
    })

  targetParser.model.copyParamsOf(referenceParser.model)

  val trainer = LHRTransferLearning(
    referenceParser = referenceParser,
    targetParser = targetParser,
    epochs = parsedArgs.epochs,
    validator = Validator(neuralParser = targetParser, goldFilePath = parsedArgs.validationSetPath),
    modelFilename = parsedArgs.modelPath,
    verbose = !parsedArgs.quiet)

  println("\n-- MODEL")
  println(targetParser.model)

  println("\n-- START TRAINING ON %d SENTENCES".format(trainingSentences.size))
  println(trainer)

  trainer.train(trainingSentences = trainingSentences)
}

/**
 * @param model the parser model to copy
 */
fun LHRModel.copyParamsOf(model: LHRModel) {

  this.headsEncoderModel.biRNN.model.assignValues(model.headsEncoderModel.biRNN.model)
  this.labelerModel?.networkModel?.model?.assignValues(model.labelerModel!!.networkModel.model)
  this.rootEmbedding.array.values.assignValues(model.rootEmbedding.array.values)
}

/**
 * Build the LHR Parser.
 *
 * @param parsedArgs the parsed command line arguments
 * @param corpus the corpus dictionary
 *
 * @return a new parser
 */
fun buildTargetParser(parsedArgs: TransferLearningArgs,
                      referenceModel: LHRModel,
                      corpus: CorpusDictionary) = LHRParser(model = LHRModel(
  langCode = referenceModel.langCode,
  corpusDictionary = corpus,
  tokensEncoderModel = buildTokensEncoderModel(parsedArgs, corpus),
  contextBiRNNConfig = referenceModel.contextBiRNNConfig,
  headsBiRNNConfig = referenceModel.headsBiRNNConfig,
  useLabeler = referenceModel.useLabeler,
  lossCriterionType = referenceModel.lossCriterionType,
  predictPosTags = referenceModel.predictPosTags))

/**
 * Build a tokens-encoder model
 *
 * @param parsedArgs the parsed command line arguments
 * @param corpus the corpus dictionary
 *
 * @return a new tokens-encoder model
 */
fun buildTokensEncoderModel(parsedArgs: TransferLearningArgs, corpus: CorpusDictionary): TokensEncoderModel {

  return ConcatTokensEncoderModel(models = listOf(

    EmbeddingsEncoderByDictionaryModel(
      embeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.wordEmbeddingSize,
        dictionary = corpus.words),
      tokenEmbeddingKey = WordEmbeddingsKey,
      dropoutCoefficient = parsedArgs.wordDropoutCoefficient
    ),

    EmbeddingsEncoderByDictionaryModel(
      embeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.posEmbeddingSize,
        dictionary = DictionarySet(corpus.posTags.getElements().map { it.label })),
      tokenEmbeddingKey = POSEmbeddingsKey,
      dropoutCoefficient = parsedArgs.posDropoutCoefficient))
  )
}