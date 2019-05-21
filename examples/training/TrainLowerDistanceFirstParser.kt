/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package training

import com.kotlinnlp.linguisticdescription.language.getLanguageByIso
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.*
import com.kotlinnlp.neuralparser.parsers.distance.*
import com.kotlinnlp.neuralparser.parsers.distance.decoder.LowerDistanceFirstDecoder
import com.kotlinnlp.neuralparser.parsers.distance.helpers.DistanceParserTrainer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.tokensencoder.wrapper.MirrorConverter
import com.kotlinnlp.neuralparser.parsers.lhrparser.sentenceconverters.FormConverter
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel

/**
 * Train a [DistanceParser] that uses a [LowerDistanceFirstDecoder].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val trainingSentences: List<CoNLLSentence> = loadSentences(
    type = "training",
    filePath = parsedArgs.trainingSetPath,
    maxSentences = parsedArgs.maxSentences,
    skipNonProjective = parsedArgs.skipNonProjective)

  val corpus: CorpusDictionary = trainingSentences.let {
    println("Creating corpus dictionary...")
    CorpusDictionary(it)
  }

  val parser: DistanceParser = buildParser(
    parsedArgs = parsedArgs,
    tokensEncoderModel = buildTokensEncoderModel(
      parsedArgs = parsedArgs,
      corpus = corpus))

  val trainer = buildTrainer(parser = parser, parsedArgs = parsedArgs)

  println("\n-- MODEL")
  println(parser.model)

  println("\n-- START TRAINING ON %d SENTENCES".format(trainingSentences.size))
  println(trainer)

  trainer.train(trainingSentences = trainingSentences)
}

/**
 * Build the LHR Parser.
 *
 * @param parsedArgs the parsed command line arguments
 * @return a new parser
 */
private fun buildParser(
  parsedArgs: CommandLineArguments,
  tokensEncoderModel: TokensEncoderWrapperModel<ParsingToken, ParsingSentence, *, *>
) = DistanceParser(
  model = DistanceParserModel(
    language = getLanguageByIso(parsedArgs.langCode),
    tokensEncoderModel = tokensEncoderModel,
    contextBiRNNConfig = BiRNNConfig(
      connectionType = LayerType.Connection.LSTM,
      hiddenActivation = Tanh(),
      numberOfLayers = parsedArgs.numOfContextLayers),
    decoderClass = LowerDistanceFirstDecoder::class))

/**
 * Build a tokens-encoder wrapper model.
 *
 * @param parsedArgs the parsed command line arguments
 * @param corpus the corpus dictionary
 *
 * @return a new tokens-encoder wrapper model
 */
private fun buildTokensEncoderModel(
  parsedArgs: CommandLineArguments,
  corpus: CorpusDictionary
): TokensEncoderWrapperModel<ParsingToken, ParsingSentence, *, *> {

  val embeddingsMap = EmbeddingsMap.fromSet(
    size = parsedArgs.wordEmbeddingSize,
    elements = corpus.words.getElementsReversedSet())

  return TokensEncoderWrapperModel(
    EnsembleTokensEncoderModel.ComponentModel(
      TokensEncoderWrapperModel(
        model = EmbeddingsEncoderModel.Base(
          embeddingsMap = embeddingsMap,
          embeddingKeyExtractor = NormWordKeyExtractor(),
          frequencyDictionary = corpus.words.getElements().associate { it to corpus.words.getCount(it) },
          dropout = parsedArgs.wordDropoutCoefficient),
        converter = FormConverter()),
      trainable = true),
    converter = MirrorConverter())

  /*
  val preEmbeddingsMap = parsedArgs.embeddingsPath!!.let {
    println("Loading pre-trained word embeddings from '$it'...")
    EmbeddingsMap.load(filename = it)
  }

  return TokensEncoderWrapperModel(
    model = EnsembleTokensEncoderModel(
      components = listOf(
        EnsembleTokensEncoderModel.ComponentModel(
          TokensEncoderWrapperModel(
            model = EmbeddingsEncoderModel.Base(
              embeddingsMap = preEmbeddingsMap,
              embeddingKeyExtractor = WordKeyExtractor(),
              fallbackEmbeddingKeyExtractors = listOf(NormWordKeyExtractor()),
              dropout = 0.0),
            converter = FormConverter()),
          trainable = false),
        EnsembleTokensEncoderModel.ComponentModel(
          TokensEncoderWrapperModel(
            model = EmbeddingsEncoderModel.Base(
              embeddingsMap = embeddingsMap,
              embeddingKeyExtractor = NormWordKeyExtractor(),
              frequencyDictionary = corpus.words.getElements().associate { it to corpus.words.getCount(it) },
              dropout = parsedArgs.wordDropoutCoefficient),
            converter = FormConverter()),
          trainable = true)),
      outputMergeConfiguration = ConcatMerge()),
    converter = MirrorConverter()
  )
  */
}

/**
 * Build a trainer.
 *
 * @param parser the parser
 * @param parsedArgs the parsed command line arguments
 *
 * @return a trainer for the given [parser]
 */
private fun buildTrainer(parser: DistanceParser,
                         parsedArgs: CommandLineArguments): DistanceParserTrainer {

  val preprocessor: SentencePreprocessor = BasePreprocessor()

  return DistanceParserTrainer(
    parser = parser,
    epochs = parsedArgs.epochs,
    batchSize = parsedArgs.batchSize,
    validator = null,
    modelFilename = parsedArgs.modelPath,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    sentencePreprocessor = preprocessor,
    verbose = !parsedArgs.quiet)
}
