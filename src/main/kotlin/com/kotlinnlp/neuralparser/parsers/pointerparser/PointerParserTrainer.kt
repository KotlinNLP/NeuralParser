/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.pointerparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.helpers.Trainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork.PointerNetworkProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.assignSum
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling
import com.kotlinnlp.utils.stats.MovingAverage

/**
 * The training helper.
 *
 * @param parser a neural parser
 * @param batchSize the size of the batches of sentences
 * @param epochs the number of training epochs
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param updateMethod the update method shared to all the parameters of the parser (Learning Rate, ADAM, AdaGrad, ...)
 * @param sentencePreprocessor the sentence preprocessor (e.g. to perform morphological analysis)
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class PointerParserTrainer(
  private val parser: PointerParser,
  private val batchSize: Int,
  private val epochs: Int,
  validator: Validator?,
  modelFilename: String,
  private val updateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
  sentencePreprocessor: SentencePreprocessor = BasePreprocessor(),
  verbose: Boolean = true
) : Trainer(
  neuralParser = parser,
  batchSize = batchSize,
  epochs = epochs,
  validator = validator,
  modelFilename = modelFilename,
  minRelevantErrorsCountToUpdate = 1,
  sentencePreprocessor = sentencePreprocessor,
  verbose = verbose
) {

  /**
   * The tokens encoder.
   */
  private val tokensEncoderWrapper = this.parser.model.tokensEncoderModel.buildEncoder(useDropout = false)

  /**
   * The encoder of tokens encodings sentential context.
   */
  private val contextEncoder = DeepBiRNNEncoder<DenseNDArray>(
    network = this.parser.model.contextEncoderModel,
    propagateToInput = true,
    useDropout = false)

  /**
   * The positional encoder.
   */
  private val depsPointer = PointerNetworkProcessor(this.parser.model.depsPointerNetworkModel)

  /**
   * The positional encoder.
   */
  private val headsPointer = PointerNetworkProcessor(this.parser.model.headsPointerNetworkModel)

  /**
   * The labeler.
   */
  private val labeler = Labeler(
    model = this.parser.model.labelerNetworkModel,
    grammaticalConfigurations = this.parser.model.grammaticalConfigurations,
    useDropout = false)

  /**
   * The optimizer of the tokens encoder.
   */
  private val optimizer = ParamsOptimizer(this.updateMethod)

  /**
   * The epoch counter.
   */
  private var epochCount: Int = 0

  /**
   *
   */
  private val positiveClassAvg = MovingAverage()

  /**
   * @return a string representation of the configuration of this Trainer
   */
  override fun toString(): String = """
    %-33s : %s
    %-33s : %s
  """.trimIndent().format(
    "Epochs", this.epochs,
    "Batch size", this.batchSize
  )

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    if (this.updateMethod is BatchScheduling) this.updateMethod.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {

    if (this.updateMethod is EpochScheduling) this.updateMethod.newEpoch()

    this.epochCount++
  }

  /**
   * Update the model parameters.
   */
  override fun update() = this.optimizer.update()

  /**
   * @return the count of the relevant errors
   */
  override fun getRelevantErrorsCount(): Int = 1

  /**
   * Method to call before learning a new sentence.
   */
  private fun beforeSentenceLearning() {
    if (this.updateMethod is ExampleScheduling) this.updateMethod.newExample()
  }

  /**
   * Train the Transition System with the given [sentence] and [goldTree].
   *
   * @param sentence the sentence
   * @param goldTree the gold tree of the sentence
   */
  override fun trainSentence(sentence: ParsingSentence, goldTree: DependencyTree) {

    this.beforeSentenceLearning()

    val tokensEncodings: List<DenseNDArray> = this.tokensEncoderWrapper.forward(sentence)
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings).map { it.copy() }

    val tokensDependents: List<DenseNDArray> = this.depsPointer.forward(contextVectors)
    val tokensDependentsErrors: List<DenseNDArray> = this.calculateDepsErrors(tokensDependents, goldTree)

    val tokensHeads: List<DenseNDArray> = this.headsPointer.forward(contextVectors)
    val tokensHeadsErrors: List<DenseNDArray> = this.calculateHeadsErrors(tokensHeads, goldTree)

    val tokensLabels: List<DenseNDArray> = this.labeler.forward(Labeler.Input(sentence, contextVectors, goldTree))
    val tokensLabelsErrors: List<DenseNDArray> = this.calculateLabelerLoss(tokensLabels, goldTree)

    this.propagateErrors(
      sentenceLength = sentence.tokens.size,
      depsOutputErrors = tokensDependentsErrors,
      headsOutputErrors = tokensHeadsErrors,
      labelerOutputErrors = tokensLabelsErrors)
  }

  /**
   *
   */
  private fun propagateErrors(sentenceLength: Int,
                              depsOutputErrors: List<DenseNDArray>?,
                              headsOutputErrors: List<DenseNDArray>?,
                              labelerOutputErrors: List<DenseNDArray>?) {

    val tokensEncodingsErrors: List<DenseNDArray> = List(size = sentenceLength, init = {
      DenseNDArrayFactory.zeros(Shape(this.parser.model.tokenEncodingSize))
    })

    val contextVectorsErrors: List<DenseNDArray> = List(size = sentenceLength, init = {
      DenseNDArrayFactory.zeros(Shape(this.parser.model.contextVectorsSize))
    })

    if (depsOutputErrors != null) {
      this.depsPointer.propagateErrors(depsOutputErrors, optimizer = this.optimizer, copy = false).let {
        contextVectorsErrors.assignSum(it.inputVectorsErrors)
        contextVectorsErrors.assignSum(it.inputSequenceErrors)
      }
    }

    if (headsOutputErrors != null) {
      this.headsPointer.propagateErrors(headsOutputErrors, optimizer = this.optimizer, copy = false).let {
        contextVectorsErrors.assignSum(it.inputVectorsErrors)
        contextVectorsErrors.assignSum(it.inputSequenceErrors)
      }
    }

    if (labelerOutputErrors != null) {
      this.labeler.propagateErrors(labelerOutputErrors, optimizer = this.optimizer, copy = false).let {
        contextVectorsErrors.assignSum(it)
      }
    }

    this.contextEncoder.propagateErrors(contextVectorsErrors, optimizer = this.optimizer, copy = false).let {
      tokensEncodingsErrors.assignSum(it)
    }

    this.tokensEncoderWrapper.propagateErrors(tokensEncodingsErrors, optimizer = this.optimizer)
  }

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  private fun PointerNetworkProcessor.forward(input: List<DenseNDArray>): List<DenseNDArray> {

    this.setInputSequence(input)

    return input.map { this.forward(it) }
  }

  /**
   * @param predictions the list of prediction
   * @param dependencyTree the gold tree of the sentence
   *
   * @return the errors of the given predictions
   */
  private fun calculateDepsErrors(predictions: List<DenseNDArray>, dependencyTree: DependencyTree): List<DenseNDArray> {

    return predictions.mapIndexed { tokenIndex, prediction ->

      val expectedValues = DenseNDArrayFactory.fill(Shape(predictions.size), 0.0)

      dependencyTree.getDependents(dependencyTree.elements[tokenIndex]).forEach { depId ->
        expectedValues[dependencyTree.getPosition(depId)] = 1.0

        this.positiveClassAvg.add(prediction[dependencyTree.getPosition(depId)])
      }

      MSECalculator().calculateErrors(output = prediction, outputGold = expectedValues)
    }
  }

  /**
   * @param predictions the list of prediction
   * @param dependencyTree the gold tree of the sentence
   *
   * @return the errors of the given predictions
   */
  private fun calculateHeadsErrors(predictions: List<DenseNDArray>, dependencyTree: DependencyTree): List<DenseNDArray> {

    return predictions.mapIndexed { tokenIndex, prediction ->

      val expectedValues = DenseNDArrayFactory.fill(Shape(predictions.size), 0.0)

      dependencyTree.getHead(dependencyTree.elements[tokenIndex])?.let { headId ->
        expectedValues[dependencyTree.getPosition(headId)] = 1.0
      }

      MSECalculator().calculateErrors(output = prediction, outputGold = expectedValues)
    }
  }

  /**
   * Return the errors of a given labeler predictions, respect to a gold dependency tree.
   * Errors are calculated comparing the last predictions done with the given gold grammatical configurations.
   *
   * @param predictions the current network predictions
   * @param dependencyTree the gold tree of the sentence
   *
   * @return a list of predictions errors
   */
  private fun calculateLabelerLoss(predictions: List<DenseNDArray>, dependencyTree: DependencyTree): List<DenseNDArray> {

    return predictions.mapIndexed { tokenIndex, prediction ->

      val tokenId: Int = dependencyTree.elements[tokenIndex]

      SoftmaxCrossEntropyCalculator().calculateErrors(
        output = prediction,
        goldIndex = this.parser.model.grammaticalConfigurations.getId(dependencyTree.getConfiguration(tokenId)!!)!!)

    }
  }
}
