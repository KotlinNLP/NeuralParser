/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.contextencoder.ContextEncoderOptimizer
import com.kotlinnlp.neuralparser.helpers.Trainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizerFactory
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapper

/**
 * The transfer learning training helper.
 *
 * @param referenceParser the neural parser used as reference
 * @param targetParser the neural parser to train via transfer learning
 * @param epochs the number of training epochs
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param sentencePreprocessor the sentence preprocessor (e.g. to perform morphological analysis)
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class LHRTransferLearning(
  private val referenceParser: LHRParser,
  private val targetParser: LHRParser,
  private val epochs: Int,
  validator: Validator?,
  modelFilename: String,
  private val updateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  sentencePreprocessor: SentencePreprocessor = BasePreprocessor(),
  verbose: Boolean = true
) : Trainer(
  neuralParser = targetParser,
  batchSize = 1,
  epochs = epochs,
  validator = validator,
  modelFilename = modelFilename,
  minRelevantErrorsCountToUpdate = 1,
  sentencePreprocessor = sentencePreprocessor,
  verbose = verbose
) {

  /**
   * The [TokensEncoder] of the reference parser.
   */
  private val referenceTokensEncoder: TokensEncoderWrapper<ParsingToken, ParsingSentence, *, *> =
    this.referenceParser.model.tokensEncoderWrapperModel.buildWrapper(useDropout = false)

  /**
   * The [ContextEncoder] of the reference parser.
   */
  private val referenceContextEncoder = ContextEncoder(
    model = this.referenceParser.model.contextEncoderModel,
    useDropout = false)

  /**
   * The [TokensEncoder] of the target parser.
   */
  private val targetTokensEncoder: TokensEncoderWrapper<ParsingToken, ParsingSentence, *, *> =
    this.targetParser.model.tokensEncoderWrapperModel.buildWrapper(useDropout = true)

  /**
   * The [ContextEncoder] of the target parser.
   */
  private val targetContextEncoder = ContextEncoder(
    this.targetParser.model.contextEncoderModel, useDropout = true)

  /**
   * The optimizer of the context encoder.
   */
  private val targetContextEncoderOptimizer = ContextEncoderOptimizer(
    model = this.targetParser.model.contextEncoderModel, updateMethod = this.updateMethod)

  /**
   * The optimizer of the tokens encoder.
   */
  private val targetTokensEncoderOptimizer = TokensEncoderOptimizerFactory(
    model = this.targetParser.model.tokensEncoderWrapperModel.model, updateMethod = this.updateMethod)

  /**
   * Train the Transition System with the given [sentence] and [goldTree].
   * Transfer the knowledge acquired by the context encoder of a reference parser to that of the target parser.
   *
   * @param sentence the sentence
   * @param goldTree the gold tree of the sentence
   */
  override fun trainSentence(sentence: ParsingSentence, goldTree: DependencyTree) {

    this.beforeSentenceLearning()

    val targetTokensEncodings: List<DenseNDArray> = this.targetTokensEncoder.forward(sentence)
    val targetContextVectors = this.targetContextEncoder.forward(targetTokensEncodings)

    val refTokensEncodings: List<DenseNDArray> = this.referenceTokensEncoder.forward(sentence)
    val refContextVectors = this.referenceContextEncoder.forward(refTokensEncodings)

    val errors: List<DenseNDArray> = MSECalculator().calculateErrors(
      outputSequence = targetContextVectors,
      outputGoldSequence = refContextVectors)

    this.targetTokensEncoder.propagateErrors(this.targetContextEncoder.propagateErrors(errors))
  }

  /**
   * Propagate the [outputErrors] through the target context encoder, accumulates the resulting parameters errors in the
   * [targetContextEncoderOptimizer] and returns the input errors.
   *
   * @param outputErrors the output errors
   *
   * @return the input errors
   */
  private fun ContextEncoder.propagateErrors(outputErrors: List<DenseNDArray>): List<DenseNDArray> {

    this.backward(outputErrors)

    this@LHRTransferLearning.targetContextEncoderOptimizer.accumulate(this.getParamsErrors(copy = false))

    return this.getInputErrors(copy = false)
  }

  /**
   * Propagate the [outputErrors] through the tokens encoder, accumulates the resulting parameters errors in the
   * [targetTokensEncoderOptimizer] and returns the input errors.
   *
   * @param outputErrors the output errors
   */
  private fun TokensEncoderWrapper<*, *, *, *>.propagateErrors(outputErrors: List<DenseNDArray>) {

    this.backward(outputErrors)

    this@LHRTransferLearning.targetTokensEncoderOptimizer.accumulate(this.getParamsErrors(copy = false))
  }

  /**
   * Method to call before learning a new sentence.
   */
  private fun beforeSentenceLearning() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }

  /**
   * Update the model parameters.
   */
  override fun update() {
    this.targetTokensEncoderOptimizer.update()
    this.targetContextEncoderOptimizer.update()
  }

  /**
   * @return the count of the relevant errors
   */
  override fun getRelevantErrorsCount(): Int = 1

  /**
   * @return a string representation of the configuration of this Trainer
   */
  override fun toString(): String = """
    %-33s : %s
  """.trimIndent().format(
    "Epochs", this.epochs
  )
}
