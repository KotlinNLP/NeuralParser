/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package lhrparser

import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRTrainer
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils.LossCriterionType
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.keyextractors.PosTagKeyExtractor
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.keyextractors.WordKeyExtractor
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.simplednn.core.embeddings.EMBDLoader
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.AffineMerge
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.morpho.FeaturesCollector
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderModel
import com.kotlinnlp.utils.DictionarySet
import lhrparser.utils.TrainingArgs
import java.io.File
import java.io.FileInputStream

/**
 * Train the [LHRParser].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = TrainingArgs(args)

  val trainingSentences: List<CoNLLSentence> = loadSentences(
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
 * Build the LHR Parser.
 *
 * @param parsedArgs the parsed command line arguments
 * @param tokensEncoderModel the tokens-encoder model
 * @param corpus the corpus dictionary
 *
 * @return a new parser
 */
private fun buildParser(parsedArgs: TrainingArgs,
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

/**
 * Build a tokens-encoder model.
 *
 * @param parsedArgs the parsed command line arguments
 * @param corpus the corpus dictionary
 *
 * @return a new tokens-encoder model
 */
private fun buildTokensEncoderModel(parsedArgs: TrainingArgs,
                                    sentences: List<CoNLLSentence>, // TODO: it will be used to initialize the MorphoEncoder
                                    corpus: CorpusDictionary): TokensEncoderModel =

  when (parsedArgs.tokensEncodingType) {

    TrainingArgs.TokensEncodingType.WORD_AND_EXT_AND_POS_EMBEDDINGS -> { // TODO: separate with a dedicated builder

      val embeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.wordEmbeddingSize,
        dictionary = corpus.words)

      val preEmbeddingsMap = parsedArgs.embeddingsPath!!.let {
        println("Loading pre-trained word embeddings from '$it'...")
        EMBDLoader().load(filename = it)
      }

      val posEmbeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.posEmbeddingSize,
        dictionary = DictionarySet(corpus.posTags.getElements().map { it.label }))

      EnsembleTokensEncoderModel(models = listOf(
        EmbeddingsEncoderModel(embeddingsMap = preEmbeddingsMap,
          embeddingKeyExtractor = WordKeyExtractor,
          dropoutCoefficient = parsedArgs.wordDropoutCoefficient),
        EmbeddingsEncoderModel(embeddingsMap = embeddingsMap,
          embeddingKeyExtractor = WordKeyExtractor,
          dropoutCoefficient = parsedArgs.wordDropoutCoefficient),
        EmbeddingsEncoderModel(embeddingsMap = posEmbeddingsMap,
          embeddingKeyExtractor = PosTagKeyExtractor,
          dropoutCoefficient = parsedArgs.posDropoutCoefficient)),
        outputMergeConfiguration = AffineMerge(
          outputSize = 100, // TODO
          activationFunction = null))
    }

    TrainingArgs.TokensEncodingType.WORD_AND_POS_EMBEDDINGS -> { // TODO: separate with a dedicated builder

      val embeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.wordEmbeddingSize,
        dictionary = corpus.words)

      val posEmbeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.posEmbeddingSize,
        dictionary = DictionarySet(corpus.posTags.getElements().map { it.label }))

      EnsembleTokensEncoderModel(models = listOf(
        EmbeddingsEncoderModel(embeddingsMap = embeddingsMap,
          embeddingKeyExtractor = WordKeyExtractor,
          dropoutCoefficient = parsedArgs.wordDropoutCoefficient),
        EmbeddingsEncoderModel(embeddingsMap = posEmbeddingsMap,
          embeddingKeyExtractor = PosTagKeyExtractor,
          dropoutCoefficient = parsedArgs.posDropoutCoefficient)),
        outputMergeConfiguration = AffineMerge(
          outputSize = 100, // TODO
          activationFunction = null))
    }

    TrainingArgs.TokensEncodingType.WORD_EMBEDDINGS -> { // TODO: separate with a dedicated builder

      val embeddingsMap = EmbeddingsMapByDictionary(
        size = parsedArgs.wordEmbeddingSize,
        dictionary = corpus.words)

      EmbeddingsEncoderModel(embeddingsMap = embeddingsMap,
        embeddingKeyExtractor = WordKeyExtractor,
        dropoutCoefficient = parsedArgs.wordDropoutCoefficient)
    }

    TrainingArgs.TokensEncodingType.MORPHO_FEATURES -> {

      val morphoAnalyzer: MorphologicalAnalyzer = parsedArgs.morphoDictionaryPath.let {

        println("Loading serialized dictionary from '$it'...")

        MorphologicalAnalyzer(
          langCode = parsedArgs.langCode,
          dictionary = MorphologyDictionary.load(FileInputStream(File(it))))
      }

      val lexiconDictionary = LexiconDictionary.load(parsedArgs.lexiconDictionaryPath!!)

      val featuresDictionary = FeaturesCollector(
        lexicalDictionary = lexiconDictionary,
        sentences = sentences.map {

          /*
          TODO: under development

          val analysis = morphoAnalyzer.analyze(sentence = it as RealSentence<RealToken>)
          analysis.tokens
          */

        }).collect()

      MorphoEncoderModel(
        lexiconDictionary = lexiconDictionary,
        featuresDictionary = featuresDictionary,
        tokenEncodingSize = parsedArgs.wordEmbeddingSize,
        activation = null)
    }
  }

/**
 * Build a trainer for a given [LHRParser].
 *
 * @param parser an LHR parser
 * @param parsedArgs the parsed command line arguments
 *
 * @return a trainer for the given [parser]
 */
private fun buildTrainer(parser: LHRParser, parsedArgs: TrainingArgs) = LHRTrainer(
  parser = parser,
  epochs = parsedArgs.epochs,
  batchSize = parsedArgs.batchSize,
  validator = Validator(
    neuralParser = parser,
    sentences = loadSentences(
      type = "validation",
      filePath = parsedArgs.validationSetPath,
      maxSentences = null,
      skipNonProjective = false)),
  modelFilename = parsedArgs.modelPath,
  lhrErrorsOptions = LHRTrainer.LHRErrorsOptions(
    skipPunctuationErrors = parsedArgs.skipPunctuationErrors,
    usePositionalEncodingErrors = false),
  updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  verbose = !parsedArgs.quiet)
