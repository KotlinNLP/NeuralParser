/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package training

import buildSentencePreprocessor
import com.kotlinnlp.languagemodel.CharLM
import com.kotlinnlp.linguisticdescription.language.getLanguageByIso
import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.linguisticdescription.morphology.MorphologicalAnalysis
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.neuralparser.helpers.validator.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.*
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.xenomachina.argparser.mainBody
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRTrainer
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.LossCriterionType
import com.kotlinnlp.tokensencoder.wrapper.MirrorConverter
import com.kotlinnlp.neuralparser.parsers.lhrparser.sentenceconverters.MorphoConverter
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.keyextractors.PosTagKeyExtractor
import com.kotlinnlp.neuralparser.parsers.lhrparser.sentenceconverters.FormConverter
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.NormWordKeyExtractor
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.AffineMerge
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatMerge
import com.kotlinnlp.tokensencoder.charlm.CharLMEncoderModel
import com.kotlinnlp.tokensencoder.ensemble.EnsembleTokensEncoderModel
import com.kotlinnlp.tokensencoder.morpho.FeaturesCollector
import com.kotlinnlp.tokensencoder.morpho.MorphoEncoderModel
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel
import java.io.File
import java.io.FileInputStream

/**
 * Train the [LHRParser].
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

  val morphologyDictionary: MorphologyDictionary? = parsedArgs.morphoDictionaryPath?.let {
    println("Loading serialized dictionary from '$it'...")
    MorphologyDictionary.load(FileInputStream(File(it)))
  }

  val parser: LHRParser = buildParser(
    parsedArgs = parsedArgs,
    tokensEncoderWrapperModel = buildTokensEncoderWrapperModel(
      parsedArgs = parsedArgs,
      sentences = trainingSentences,
      corpus = corpus,
      morphologyDictionary = morphologyDictionary),
    corpus = corpus)

  val trainer = buildTrainer(parser = parser, parsedArgs = parsedArgs, morphologyDictionary = morphologyDictionary)

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
 * @param tokensEncoderWrapperModel the tokens-encoder wrapper model
 * @param corpus the corpus dictionary
 *
 * @return a new parser
 */
private fun buildParser(
  parsedArgs: CommandLineArguments,
  tokensEncoderWrapperModel: TokensEncoderWrapperModel<ParsingToken, ParsingSentence, *, *>,
  corpus: CorpusDictionary
): LHRParser = LHRParser(model = LHRModel(
  corpusDictionary = corpus,
  lssModel = LSSModel(
    language = getLanguageByIso(parsedArgs.langCode),
    tokensEncoderWrapperModel = tokensEncoderWrapperModel,
    contextBiRNNConfig = BiRNNConfig(
      connectionType = LayerType.Connection.LSTM,
      hiddenActivation = Tanh,
      numberOfLayers = parsedArgs.numOfContextLayers),
    headsBiRNNConfig = BiRNNConfig(
      connectionType = LayerType.Connection.LSTM,
      hiddenActivation = Tanh)
  ),
  useLabeler = !parsedArgs.noLabeler,
  lossCriterionType = LossCriterionType.Softmax,
  predictPosTags = !parsedArgs.noPosPrediction))

/**
 * Build a tokens-encoder wrapper model.
 *
 * @param parsedArgs the parsed command line arguments
 * @param corpus the corpus dictionary
 *
 * @return a new tokens-encoder wrapper model
 */
private fun buildTokensEncoderWrapperModel(
  parsedArgs: CommandLineArguments,
  sentences: List<CoNLLSentence>, // TODO: it will be used to initialize the MorphoEncoder
  corpus: CorpusDictionary,
  morphologyDictionary: MorphologyDictionary?
): TokensEncoderWrapperModel<ParsingToken, ParsingSentence, *, *> =

  when (parsedArgs.tokensEncodingType) {

    CommandLineArguments.TokensEncodingType.WORD_AND_EXT_AND_POS_EMBEDDINGS -> { // TODO: separate with a dedicated builder

      val embeddingsMap = EmbeddingsMap.fromSet(
        size = parsedArgs.wordEmbeddingSize,
        elements = corpus.words.getElementsReversedSet())

      val preEmbeddingsMap = parsedArgs.embeddingsPath!!.let {
        println("Loading pre-trained word embeddings from '$it'...")
        EmbeddingsMap.load(filename = it)
      }

      val posEmbeddingsMap = EmbeddingsMap.fromSet(
        size = parsedArgs.posEmbeddingSize,
        elements = corpus.grammaticalConfigurations.getElements().mapNotNull { it.posToString }.toSet())

      TokensEncoderWrapperModel(
        model = EnsembleTokensEncoderModel(
          components = listOf(
            EnsembleTokensEncoderModel.ComponentModel(
              model = TokensEncoderWrapperModel(
                model = EmbeddingsEncoderModel.Base(
                  embeddingsMap = preEmbeddingsMap,
                  embeddingKeyExtractor = NormWordKeyExtractor(),
                  dropout = parsedArgs.wordDropoutCoefficient),
                converter = FormConverter()),
              trainable = true),
            EnsembleTokensEncoderModel.ComponentModel(
              model = TokensEncoderWrapperModel(
                model = EmbeddingsEncoderModel.Base(
                  embeddingsMap = embeddingsMap,
                  embeddingKeyExtractor = NormWordKeyExtractor(),
                  frequencyDictionary = corpus.words.getElements().associateWith { corpus.words.getCount(it) },
                  dropout = parsedArgs.wordDropoutCoefficient),
                converter = FormConverter()),
              trainable = true),
            EnsembleTokensEncoderModel.ComponentModel(
              model = TokensEncoderWrapperModel(
                model = EmbeddingsEncoderModel.Base(
                  embeddingsMap = posEmbeddingsMap,
                  embeddingKeyExtractor = PosTagKeyExtractor,
                  frequencyDictionary = corpus.grammaticalConfigurations.getElements()
                    .mapNotNull { it.posToString }
                    .associateWith { 1 },
                  dropout = parsedArgs.posDropoutCoefficient),
                converter = MirrorConverter()),
              trainable = true)
          ),
          outputMergeConfiguration = AffineMerge(
            outputSize = 100, // TODO
            activationFunction = null)),
        converter = MirrorConverter()
      )
    }

    CommandLineArguments.TokensEncodingType.WORD_AND_POS_EMBEDDINGS -> { // TODO: separate with a dedicated builder

      val embeddingsMap = EmbeddingsMap.fromSet(
        size = parsedArgs.wordEmbeddingSize,
        elements = corpus.words.getElementsReversedSet())

      val posEmbeddingsMap = EmbeddingsMap.fromSet(
        size = parsedArgs.posEmbeddingSize,
        elements = corpus.grammaticalConfigurations.getElements().mapNotNull { it.posToString }.toSet())

      TokensEncoderWrapperModel(
        model = EnsembleTokensEncoderModel(
          components = listOf(
            EnsembleTokensEncoderModel.ComponentModel(
              model = TokensEncoderWrapperModel(
                model = EmbeddingsEncoderModel.Base(
                  embeddingsMap = embeddingsMap,
                  embeddingKeyExtractor = NormWordKeyExtractor(),
                  frequencyDictionary = corpus.words.getElements().associateWith { corpus.words.getCount(it) },
                  dropout = parsedArgs.wordDropoutCoefficient),
                converter = FormConverter()),
              trainable = true),
            EnsembleTokensEncoderModel.ComponentModel(
              model = TokensEncoderWrapperModel(
                model = EmbeddingsEncoderModel.Base(
                  embeddingsMap = posEmbeddingsMap,
                  embeddingKeyExtractor = PosTagKeyExtractor,
                  frequencyDictionary = corpus.grammaticalConfigurations.getElements()
                    .mapNotNull { it.posToString }
                    .associateWith { 1 },
                  dropout = parsedArgs.posDropoutCoefficient),
                converter = MirrorConverter()),
              trainable = true)
          ),
          outputMergeConfiguration = ConcatMerge()),
        converter = MirrorConverter()
      )
    }

    CommandLineArguments.TokensEncodingType.WORD_EMBEDDINGS -> { // TODO: separate with a dedicated builder

      val embeddingsMap = EmbeddingsMap.fromSet(
        size = parsedArgs.wordEmbeddingSize,
        elements = corpus.words.getElementsReversedSet())

      TokensEncoderWrapperModel(
        model = EmbeddingsEncoderModel.Base(
          embeddingsMap = embeddingsMap,
          embeddingKeyExtractor = NormWordKeyExtractor(),
          frequencyDictionary = corpus.words.getElements().associateWith { corpus.words.getCount(it) },
          dropout = parsedArgs.wordDropoutCoefficient),
        converter = FormConverter()
      )
    }

    CommandLineArguments.TokensEncodingType.CHARLM -> { // TODO: separate with a dedicated builder
      TokensEncoderWrapperModel(
        model = CharLMEncoderModel(
          dirCharLM = CharLM.load(File(parsedArgs.charLMModelPath!!).inputStream()),
          revCharLM = CharLM.load(File(parsedArgs.charLMRevModelPath!!).inputStream()),
          outputMergeConfiguration = AffineMerge(
            outputSize = 100, // TODO
            activationFunction = Tanh)),
        converter = FormConverter()
      )
    }

    CommandLineArguments.TokensEncodingType.MORPHO_FEATURES -> {

      val analyzer = MorphologicalAnalyzer(dictionary = morphologyDictionary!!)

      val lexiconDictionary = parsedArgs.lexiconDictionaryPath?.let {
        println("Loading lexicon from '$it'...")
        LexiconDictionary.load(it)
      }

      val featuresDictionary = FeaturesCollector(
        lexicalDictionary = lexiconDictionary,
        sentences = sentences.mapIndexed { i, it -> it.toMorphoSentence(index = i, analyzer = analyzer)}
      ).collect()

      TokensEncoderWrapperModel(
        model = MorphoEncoderModel(
          lexiconDictionary = lexiconDictionary,
          featuresDictionary = featuresDictionary,
          tokenEncodingSize = parsedArgs.wordEmbeddingSize,
          activation = null),
        converter = MorphoConverter()
      )
    }
  }


/**
 * Build a [MorphoSentence] from this [CoNLLSentence].
 *
 * @param index the position index of this sentence
 * @param analyzer a morphological analyzer
 *
 * @return a new morpho sentence
 */
private fun CoNLLSentence.toMorphoSentence(index: Int, analyzer: MorphologicalAnalyzer): MorphoSentence<FormToken> {

  val baseTokens = this.tokens.toBaseTokens()
  val position = Position(
    index = index,
    start = baseTokens.first().position.start,
    end = baseTokens.last().position.end)
  @Suppress("UNCHECKED_CAST")
  val sentence = BaseSentence(id = index, position = position, tokens = baseTokens) as RealSentence<RealToken>

  val analysis = analyzer.analyze(sentence)

  return object : MorphoSentence<FormToken> {
    override val tokens: List<FormToken> = this@toMorphoSentence.tokens
    override val morphoAnalysis: MorphologicalAnalysis? = analysis
  }
}

/**
 * Build a trainer for a given [LHRParser].
 *
 * @param parser an LHR parser
 * @param parsedArgs the parsed command line arguments
 * @param morphologyDictionary a morphology dictionary
 *
 * @return a trainer for the given [parser]
 */
private fun buildTrainer(parser: LHRParser,
                         parsedArgs: CommandLineArguments,
                         morphologyDictionary: MorphologyDictionary?): LHRTrainer {

  val preprocessor: SentencePreprocessor = buildSentencePreprocessor(morphologyDictionary)

  return LHRTrainer(
    parser = parser,
    epochs = parsedArgs.epochs,
    batchSize = parsedArgs.batchSize,
    validator = Validator(
      neuralParser = parser,
      sentences = loadSentences(
        type = "validation",
        filePath = parsedArgs.validationSetPath,
        maxSentences = null,
        skipNonProjective = false),
      sentencePreprocessor = preprocessor),
    modelFilename = parsedArgs.modelPath,
    skipPunctuationErrors = parsedArgs.skipPunctuationErrors,
    usePositionalEncodingErrors = false,
    updateMethod = RADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
    sentencePreprocessor = preprocessor,
    verbose = !parsedArgs.quiet)
}
