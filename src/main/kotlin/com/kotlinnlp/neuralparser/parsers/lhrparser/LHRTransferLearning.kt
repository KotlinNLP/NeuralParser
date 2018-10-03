/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.lssencoder.LSSEncoder
import com.kotlinnlp.lssencoder.LSSOptimizer
import com.kotlinnlp.lssencoder.LatentSyntacticStructure
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.helpers.Trainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

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
   * The [LSSEncoder] of the reference parser.
   */
  private val referenceLSSEncoder: LSSEncoder<ParsingToken, ParsingSentence> = LSSEncoder(
    model = this.referenceParser.model.lssModel,
    useDropout = false)

  /**
   * The [LSSEncoder] of the target parser.
   */
  private val targetLSSEncoder: LSSEncoder<ParsingToken, ParsingSentence> = LSSEncoder(
    model = this.targetParser.model.lssModel,
    useDropout = true)

  /**
   * The optimizer of the context encoder.
   */
  private val targetLSSEncoderOptimizer = LSSOptimizer(
    model = this.targetParser.model.lssModel,
    updateMethod = this.updateMethod
  )

  /**
   * Train the [targetParser] with the given [sentence] and [goldTree].
   * Transfer the knowledge acquired by the LSS encoder of a reference parser to that of the target parser.
   *
   * @param sentence the input sentence
   * @param goldTree the gold tree of the sentence
   */
  override fun trainSentence(sentence: ParsingSentence, goldTree: DependencyTree) {

    this.beforeSentenceLearning()

    val targetLSS: LatentSyntacticStructure<ParsingToken, ParsingSentence> = this.targetLSSEncoder.forward(sentence)
    val refLSS: LatentSyntacticStructure<ParsingToken, ParsingSentence> = this.referenceLSSEncoder.forward(sentence)

    val tokensEncodingsSize: Int = this.targetLSSEncoder.model.tokensEncoderWrapperModel.model.tokenEncodingSize
    val tokensEncodingsErrors: List<DenseNDArray> =
      sentence.tokens.map { DenseNDArrayFactory.zeros(Shape(tokensEncodingsSize)) }

    val contextErrors: List<DenseNDArray> = MSECalculator().calculateErrors(
      outputSequence = targetLSS.contextVectors,
      outputGoldSequence = refLSS.contextVectors)

    this.targetLSSEncoder.propagateErrors(
      errors = LSSEncoder.OutputErrors(
        tokensEncodings = tokensEncodingsErrors,
        contextVectors = contextErrors,
        latentHeads = contextErrors.map { it.zerosLike() }),
      optimizer = this.targetLSSEncoderOptimizer)
  }

  /**
   * Method to call before learning a new sentence.
   */
  private fun beforeSentenceLearning() {
    if (this.updateMethod is ExampleScheduling) this.updateMethod.newExample()
  }

  /**
   * Update the model parameters.
   */
  override fun update() {
    this.targetLSSEncoderOptimizer.update()
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
