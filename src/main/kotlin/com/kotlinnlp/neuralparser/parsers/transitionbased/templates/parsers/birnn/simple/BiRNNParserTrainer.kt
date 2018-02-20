/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple

import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedTrainer
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.ActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.HighestScoreCorrectActionSelector
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.oracle.OracleFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * The training helper of the [BiRNNParser].
 *
 * @property neuralParser a neural parser
 * @param epochs the number of training epochs
 * @param batchSize the number of sentences that compose a batch
 * @param minRelevantErrorsCountToUpdate the min number of relevant errors needed to update the neural parser
 *                                       (default = 1)
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
open class BiRNNParserTrainer<StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  out SupportStructureType : DecodingSupportStructure,
  ModelType: BiRNNParserModel
  >(
  private val neuralParser: BiRNNParser<StateType, TransitionType, FeaturesErrorsType, FeaturesType,
    SupportStructureType, ModelType>,
  actionsErrorsSetter: ActionsErrorsSetter<StateType, TransitionType, DenseItem, TokensEmbeddingsContext>,
  oracleFactory: OracleFactory<StateType, TransitionType>,
  epochs: Int,
  batchSize: Int,
  minRelevantErrorsCountToUpdate: Int = 1,
  validator: Validator?,
  modelFilename: String,
  verbose: Boolean = true
) :
  TransitionBasedTrainer<
    StateType,
    TransitionType,
    TokensEmbeddingsContext,
    DenseItem,
    FeaturesErrorsType,
    FeaturesType,
    SupportStructureType,
    ModelType>
  (
    neuralParser = neuralParser,
    bestActionSelector = HighestScoreCorrectActionSelector(),
    actionsErrorsSetter = actionsErrorsSetter,
    oracleFactory = oracleFactory,
    epochs = epochs,
    batchSize = batchSize,
    minRelevantErrorsCountToUpdate = minRelevantErrorsCountToUpdate,
    validator = validator,
    modelFilename = modelFilename,
    verbose = verbose
  ) {

  /**
   * The errors to propagate to the BiRNN in case of no other errors are given for an item.
   */
  private val zerosErrors: DenseNDArray =
    DenseNDArrayFactory.zeros(Shape(this.neuralParser.model.biRNN.outputSize))

  /**
   *
   */
  private val posEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.neuralParser.model.posEmbeddings,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   *
   */
  private val wordEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.neuralParser.model.wordEmbeddings,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   * The optimizer of the deep-BiRNN.
   */
  private val deepBiRNNOptimizer = ParamsOptimizer(
    params = this.neuralParser.model.biRNN.model,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   * Callback called before the learning of a sentence.
   */
  override fun beforeSentenceLearning(context: TokensEmbeddingsContext) {

    this.deepBiRNNOptimizer.newBatch()
    this.deepBiRNNOptimizer.newExample()

    this.wordEmbeddingsOptimizer.newBatch()
    this.wordEmbeddingsOptimizer.newExample()

    this.posEmbeddingsOptimizer.newBatch()
    this.posEmbeddingsOptimizer.newExample()
  }

  /**
   * Callback called after the learning of a sentence.
   */
  override fun afterSentenceLearning(context: TokensEmbeddingsContext) {

    this.propagateErrors(context = context)
  }

  /**
   * Callback called before applying an action.
   */
  override fun beforeApplyAction(action: Transition<TransitionType, StateType>.Action,
                                 context: TokensEmbeddingsContext) = Unit

  /**
   * Update the neural components of the parser.
   */
  override fun updateNeuralComponents() {

    this.wordEmbeddingsOptimizer.update()
    this.posEmbeddingsOptimizer.update()
    this.deepBiRNNOptimizer.update()
  }

  /**
   *
   */
  private fun propagateErrors(context: TokensEmbeddingsContext) {

    this.neuralParser.biRNNEncoder.backward(
      outputErrorsSequence = context.items.map { it.errors?.array ?: this.zerosErrors }.toTypedArray(),
      propagateToInput = true)

    this.deepBiRNNOptimizer.accumulate(this.neuralParser.biRNNEncoder.getParamsErrors(copy = false))

    this.neuralParser.biRNNEncoder.getInputSequenceErrors(copy = false).forEachIndexed { i, tokenErrors ->
      this.accumulateTokenErrors(context = context, tokenIndex = i, errors = tokenErrors)
    }

    if (context.nullItemErrors != null) {

      this.neuralParser.paddingVectorEncoder.backward(
        outputErrorsSequence = arrayOf(context.nullItemErrors!!),
        propagateToInput = true)

      this.deepBiRNNOptimizer.accumulate(this.neuralParser.paddingVectorEncoder.getParamsErrors(copy = false))

      this.accumulateNullTokenErrors(this.neuralParser.paddingVectorEncoder.getInputSequenceErrors(copy = false).first())
    }
  }

  /**
   *
   */
  private fun accumulateTokenErrors(context: TokensEmbeddingsContext, tokenIndex: Int, errors: DenseNDArray) {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.neuralParser.model.posEmbeddings.size,
      this.neuralParser.model.wordEmbeddings.size)

    this.posEmbeddingsOptimizer.accumulate(
      embedding = context.posEmbeddings[tokenIndex],
      errors = splitErrors[0])

    this.wordEmbeddingsOptimizer.accumulate(
      embedding = context.wordEmbeddings[tokenIndex],
      errors = splitErrors[1])
  }

  /**
   * Accumulate the given [errors] to the embeddings of a null-token.
   *
   * @param errors the errors
   */
  private fun accumulateNullTokenErrors(errors: DenseNDArray) {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.neuralParser.model.posEmbeddings.size,
      this.neuralParser.model.wordEmbeddings.size)

    this.posEmbeddingsOptimizer.accumulate(
      embedding = this.neuralParser.model.posEmbeddings.nullEmbedding,
      errors = splitErrors[0])

    this.wordEmbeddingsOptimizer.accumulate(
      embedding = this.neuralParser.model.wordEmbeddings.nullEmbedding,
      errors = splitErrors[1])
  }
}
