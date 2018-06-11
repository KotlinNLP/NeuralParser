/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.charbased

import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedTrainer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensCharsEncodingContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.attention.han.HANEncoder
import com.kotlinnlp.simplednn.deeplearning.attention.han.HierarchySequence
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.HingeLossActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.HighestScoreCorrectActionSelector
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.oracle.OracleFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * The training helper of the [CharBasedBiRNNParser].
 *
 * @property neuralParser a neural parser
 * @param epochs the number of training epochs
 * @param batchSize the number of sentences that compose a batch
 * @param minRelevantErrorsCountToUpdate the min number of relevant errors needed to update the neural parser
 *                                       (default = 1)
 * @param learningMarginThreshold the learning margin threshold
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class CharBasedBiRNNParserTrainer<StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  out SupportStructureType : DecodingSupportStructure,
  ModelType: CharBasedBiRNNParserModel
  >(
  private val neuralParser: CharBasedBiRNNParser<StateType, TransitionType, FeaturesErrorsType, FeaturesType,
    SupportStructureType, ModelType>,
  oracleFactory: OracleFactory<StateType, TransitionType>,
  epochs: Int,
  batchSize: Int,
  minRelevantErrorsCountToUpdate: Int = 1,
  learningMarginThreshold: Double,
  validator: Validator?,
  modelFilename: String,
  verbose: Boolean = true
) :
  TransitionBasedTrainer<
    StateType,
    TransitionType,
    TokensCharsEncodingContext,
    DenseItem,
    FeaturesErrorsType,
    FeaturesType,
    SupportStructureType,
    ModelType>
  (
    neuralParser = neuralParser,
    bestActionSelector = HighestScoreCorrectActionSelector(),
    actionsErrorsSetter = HingeLossActionsErrorsSetter(learningMarginThreshold = learningMarginThreshold),
    oracleFactory = oracleFactory,
    epochs = epochs,
    batchSize = batchSize,
    minRelevantErrorsCountToUpdate = minRelevantErrorsCountToUpdate,
    validator = validator,
    modelFilename = modelFilename,
    verbose = verbose
  ) {

  /**
   *
   */
  private val zerosErrors: DenseNDArray =
    DenseNDArrayFactory.zeros(Shape(this.neuralParser.model.deepBiRNN.outputSize))

  /**
   *
   */
  private val attentionNetworkOptimizer = ParamsOptimizer(
    params = this.neuralParser.model.han.params,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   *
   */
  private val charsEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.neuralParser.model.charsEmbeddings,
    updateMethod = AdaGradMethod(learningRate = 0.1))

  /**
   *
   */
  private val wordEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.neuralParser.model.wordEmbeddings,
    updateMethod = AdaGradMethod(learningRate = 0.1))

  /**
   * The optimizer of the deep-BiRNN.
   */
  private val deepBiRNNOptimizer = ParamsOptimizer(
    params = this.neuralParser.model.deepBiRNN.model,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   * Callback called before the learning of a sentence.
   */
  override fun beforeSentenceLearning(context: TokensCharsEncodingContext) {

    this.deepBiRNNOptimizer.newBatch()
    this.deepBiRNNOptimizer.newExample()

    this.wordEmbeddingsOptimizer.newBatch()
    this.wordEmbeddingsOptimizer.newExample()

    this.charsEmbeddingsOptimizer.newBatch()
    this.charsEmbeddingsOptimizer.newExample()
  }

  /**
   * Callback called after the learning of a sentence.
   */
  override fun afterSentenceLearning(context: TokensCharsEncodingContext) {

    this.propagateErrors(context = context)
  }

  /**
   * Callback called before applying an action.
   */
  override fun beforeApplyAction(action: Transition<TransitionType, StateType>.Action,
                                 context: TokensCharsEncodingContext) = Unit

  /**
   * Update the neural components of the parser.
   */
  override fun updateNeuralComponents() {

    this.wordEmbeddingsOptimizer.update()
    this.charsEmbeddingsOptimizer.update()
    this.attentionNetworkOptimizer.update()
    this.deepBiRNNOptimizer.update()
  }

  /**
   *
   */
  private fun propagateErrors(context: TokensCharsEncodingContext) {

    this.neuralParser.biRNNEncoder.backward(
      outputErrorsSequence = context.items.map { it.errors?.array ?: this.zerosErrors },
      propagateToInput = true)

    this.deepBiRNNOptimizer.accumulate(this.neuralParser.biRNNEncoder.getParamsErrors(copy = false))

    this.neuralParser.biRNNEncoder.getInputSequenceErrors(copy = false).forEachIndexed { i, tokenErrors ->
      this.accumulateTokenErrors(context = context, tokenIndex = i, errors = tokenErrors)
    }
  }

  /**
   *
   */
  private fun accumulateTokenErrors(context: TokensCharsEncodingContext, tokenIndex: Int, errors: DenseNDArray) {

    val hanEncoder: HANEncoder<DenseNDArray> = context.usedEncoders[tokenIndex]
    val splitErrors: List<DenseNDArray> = errors.splitV(
      this.neuralParser.model.charsEncodingSize,
      this.neuralParser.model.wordEmbeddings.size)

    hanEncoder.backward(outputErrors = splitErrors[0], propagateToInput = true)

    this.attentionNetworkOptimizer.accumulate(hanEncoder.getParamsErrors())

    val charsSequenceErrors: HierarchySequence<*> = hanEncoder.getInputSequenceErrors() as HierarchySequence<*>

    charsSequenceErrors.forEachIndexed { charIndex, charsErrors ->
      this.charsEmbeddingsOptimizer.accumulate(
        embedding = context.charsEmbeddings[tokenIndex][charIndex],
        errors = charsErrors as DenseNDArray)
    }

    this.wordEmbeddingsOptimizer.accumulate(
      embedding = context.wordEmbeddings[tokenIndex],
      errors = splitErrors[1])
  }
}
