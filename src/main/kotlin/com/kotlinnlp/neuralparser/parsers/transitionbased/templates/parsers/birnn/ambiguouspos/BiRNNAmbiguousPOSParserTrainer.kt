/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.ambiguouspos

import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.parsers.transitionbased.TransitionBasedTrainer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
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
 * The training helper of the [BiRNNAmbiguousPOSParser].
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
open class BiRNNAmbiguousPOSParserTrainer<StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  SupportStructureType : DecodingSupportStructure,
  ModelType: BiRNNAmbiguousPOSParserModel
  >(
  protected val neuralParser: BiRNNAmbiguousPOSParser<StateType, TransitionType, FeaturesErrorsType, FeaturesType,
    SupportStructureType, ModelType>,
  actionsErrorsSetter: ActionsErrorsSetter<StateType, TransitionType, DenseItem, TokensAmbiguousPOSContext>,
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
    TokensAmbiguousPOSContext,
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
    DenseNDArrayFactory.zeros(Shape(this.neuralParser.model.deepBiRNN.outputSize))

  /**
   * The optimizer of the word embeddings.
   */
  private val wordEmbeddingsOptimizer = EmbeddingsOptimizer(
    embeddingsMap = this.neuralParser.model.wordEmbeddings,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   * The optimizer of the pre-trained word embeddings.
   */
  private val preTrainedEmbeddingsOptimizer = if (this.neuralParser.model.preTrainedWordEmbeddings != null)
    EmbeddingsOptimizer(
      embeddingsMap = this.neuralParser.model.preTrainedWordEmbeddings!!,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))
  else
    null

  /**
   * The optimizer of the deep-BiRNN.
   */
  private val deepBiRNNOptimizer = ParamsOptimizer(
    params = this.neuralParser.model.deepBiRNN.model,
    updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999))

  /**
   * Callback called before the learning of a sentence.
   */
  override fun beforeSentenceLearning(context: TokensAmbiguousPOSContext) {

    this.deepBiRNNOptimizer.newBatch()
    this.deepBiRNNOptimizer.newExample()

    this.wordEmbeddingsOptimizer.newBatch()
    this.wordEmbeddingsOptimizer.newExample()

    this.preTrainedEmbeddingsOptimizer?.newBatch()
    this.preTrainedEmbeddingsOptimizer?.newExample()
  }

  /**
   * Callback called after the learning of a sentence.
   */
  override fun afterSentenceLearning(context: TokensAmbiguousPOSContext) {

    this.propagateErrors(context = context)
  }

  /**
   * Callback called before applying an action.
   */
  override fun beforeApplyAction(action: Transition<TransitionType, StateType>.Action,
                                 context: TokensAmbiguousPOSContext) = Unit

  /**
   * Update the neural components of the parser.
   */
  override fun updateNeuralComponents() {

    this.deepBiRNNOptimizer.update()
    this.wordEmbeddingsOptimizer.update()
    this.preTrainedEmbeddingsOptimizer?.update()
  }

  /**
   *
   */
  protected fun propagateErrors(context: TokensAmbiguousPOSContext) {

    this.neuralParser.biRNNEncoder.backward(
      outputErrorsSequence = context.items.map { it.errors?.array ?: this.zerosErrors },
      propagateToInput = true)

    this.deepBiRNNOptimizer.accumulate(this.neuralParser.biRNNEncoder.getParamsErrors(copy = false))

    this.neuralParser.biRNNEncoder.getInputSequenceErrors(copy = false).forEachIndexed { i, tokenErrors ->
      this.accumulateTokenErrors(context = context, tokenIndex = i, errors = tokenErrors)
    }

    if (context.nullItemErrors != null) {

      this.neuralParser.paddingVectorEncoder.backward(
        outputErrorsSequence = listOf(context.nullItemErrors!!),
        propagateToInput = true)

      this.deepBiRNNOptimizer.accumulate(this.neuralParser.paddingVectorEncoder.getParamsErrors(copy = false))

      this.accumulateNullTokenErrors(
        errors = this.neuralParser.paddingVectorEncoder.getInputSequenceErrors(copy = false).first())
    }
  }

  /**
   * Accumulate the errors of a token.
   *
   * @param context the input context
   * @param errors the errors of a token
   */
  private fun accumulateTokenErrors(context: TokensAmbiguousPOSContext, tokenIndex: Int, errors: DenseNDArray) {

    val (wordEmbeddingsErrors, preTrainedEmbeddingsErrors) = this.splitTokenErrors(errors)

    this.wordEmbeddingsOptimizer.accumulate(
      embedding = context.wordEmbeddings[tokenIndex],
      errors = wordEmbeddingsErrors)

    this.preTrainedEmbeddingsOptimizer?.accumulate(
      embedding = context.preTrainedWordEmbeddings!![tokenIndex],
      errors = preTrainedEmbeddingsErrors!!)
  }

  /**
   * Accumulate the given [errors] to the embeddings of a 'null' token.
   *
   * @param errors the errors of the 'null' token
   */
  private fun accumulateNullTokenErrors(errors: DenseNDArray) {

    val (wordEmbeddingsErrors, preTrainedEmbeddingsErrors) = this.splitTokenErrors(errors)

    this.wordEmbeddingsOptimizer.accumulate(
      embedding = this.neuralParser.model.wordEmbeddings.nullEmbedding,
      errors = wordEmbeddingsErrors)

    this.preTrainedEmbeddingsOptimizer?.accumulate(
      embedding = this.neuralParser.model.wordEmbeddings.nullEmbedding,
      errors = preTrainedEmbeddingsErrors!!)
  }

  /**
   * Split the errors of a token encoding among their trainable components: word embedding, pre-trained word embedding.
   * The pre-trained word embedding errors component can be null.
   *
   * @param errors the errors of a token encoding
   *
   * @return a Pair containing the errors components (word embedding, pre-trained word embedding)
   */
  private fun splitTokenErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray?> {

    val wordEmbeddingsStartIndex: Int = this.neuralParser.model.posTagsSize
    val preTrainedEmbeddingsStartIndex: Int = wordEmbeddingsStartIndex + this.neuralParser.model.wordEmbeddingSize

    val wordEmbeddingsErrors: DenseNDArray = errors.getRange(wordEmbeddingsStartIndex, preTrainedEmbeddingsStartIndex)
    val preTrainedEmbeddingsErrors: DenseNDArray = errors.getRange(preTrainedEmbeddingsStartIndex, errors.length)

    return Pair(wordEmbeddingsErrors, preTrainedEmbeddingsErrors)
  }
}
