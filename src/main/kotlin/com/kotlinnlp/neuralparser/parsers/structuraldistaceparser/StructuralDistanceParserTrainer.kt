/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistaceparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.helpers.Trainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.helpers.treeutils.DAGNode
import com.kotlinnlp.neuralparser.helpers.treeutils.DAGNodesFactory
import com.kotlinnlp.neuralparser.helpers.treeutils.Element
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.assignSum
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling
import com.kotlinnlp.neuralparser.parsers.structuraldistaceparser.StructuralDistancePredictor.Input as SDPInput
import com.kotlinnlp.utils.combine
import kotlin.math.pow

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
class StructuralDistanceParserTrainer(
  private val parser: StructuralDistanceParser,
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
  private val tokensEncoder = this.parser.model.tokensEncoderModel.buildEncoder(useDropout = false)

  /**
   * The encoder of tokens encodings sentential context.
   */
  private val contextEncoder = DeepBiRNNEncoder<DenseNDArray>(
    network = this.parser.model.contextEncoderModel,
    propagateToInput = true,
    useDropout = false)

  /**
   * The structural distance predictor.
   */
  private val distancePredictor = StructuralDistancePredictor(
    model = this.parser.model.distanceModel,
    useDropout = false)

  /**
   * The structural depth predictor.
   */
  private val depthPredictor = StructuralDepthPredictor(
    model = this.parser.model.depthModel,
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

    if (sentence.tokens.size <= 1)
      return // skip training of sentences with a single token

    val tokensEncodings: List<DenseNDArray> = this.tokensEncoder.forward(sentence)
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings).map { it.copy() }

    val pairs: List<Pair<Int, Int>> = contextVectors.indices.toList().combine()
    val distances: List<Double> = this.distancePredictor.forward(SDPInput(hiddens = contextVectors, pairs = pairs))
    val depths: List<Double> = this.depthPredictor.forward(contextVectors)

    val treeNodes: List<DAGNode<Int>> = this.createNodes(sentence, goldTree)
    val goldDistances: List<Int> = this.getGoldDistances(treeNodes)
    val goldDepths: List<Int> = this.getGoldDepths(treeNodes)

    val distanceErrors: List<Double> = this.calcDistanceErrors(
      length = sentence.tokens.size,
      prediction = distances,
      gold = goldDistances)

    val depthErrors: List<Double> = this.calcDepthErrors(
      length = sentence.tokens.size,
      prediction = depths,
      gold = goldDepths)

    this.propagateErrors(
      sentenceLength = sentence.tokens.size,
      pairsDistanceErrors = distanceErrors,
      depthErrors = depthErrors)
  }

  /**
   * Calculate the distance errors.
   *
   * @param length the sequence length used to normalize
   * @param prediction the predicted distances
   * @param gold the gold distances
   *
   * @return the errors of each prediction
   */
  private fun calcDistanceErrors(length: Int, prediction: List<Double>, gold: List<Int>): List<Double> {

    val s2 = length.toDouble().pow(2)

    return (gold).zip(prediction).map { (g, o) -> if (o < g) -1.0/s2 else 1.0/s2 }
  }

  /**
   * Calculate the depth errors.
   *
   * @param length the sequence length used to normalize
   * @param prediction the predicted depths
   * @param gold the gold distances
   *
   * @return the errors of each prediction
   */
  private fun calcDepthErrors(length: Int, prediction: List<Double>, gold: List<Int>): List<Double> {

    val s = length.toDouble()

    return (gold).zip(prediction).map { (g, o) -> if (o < g) -1.0/s else 1.0/s }
  }

  /**
   * @param sentence the sentence
   * @param goldTree the gold dependency tree
   *
   * @return the list of gold nodes
   */
  private fun createNodes(sentence: ParsingSentence, goldTree: DependencyTree): List<DAGNode<Int>> {

    val elements = sentence.tokens.map { Element(id = it.id, label = it.form) }
    val heads = sentence.tokens.map { it.id to goldTree.getHead(it.id) }

    return DAGNodesFactory(elements, heads).newNodes()
  }

  /**
   * @param nodes the nodes
   *
   * @return the distances of all pairs of nodes
   */
  private fun getGoldDistances(nodes: List<DAGNode<Int>>): List<Int> =
    nodes.combine().map { (first, second) -> first.distance(second) }

  /**
   *
   */
  private fun getGoldDepths(nodes: List<DAGNode<Int>>): List<Int> = nodes.map { it.depth }

  /**
   * @param sentenceLength the sentence length
   * @param pairsDistanceErrors the errors of the distance for each pair
   * @param depthErrors the errors of the depth for each token
   */
  private fun propagateErrors(sentenceLength: Int, pairsDistanceErrors: List<Double>, depthErrors: List<Double>) {

    val tokensEncodingsErrors: List<DenseNDArray> = List(size = sentenceLength, init = {
      DenseNDArrayFactory.zeros(Shape(this.parser.model.tokenEncodingSize))
    })

    val contextVectorsErrors: List<DenseNDArray> = List(size = sentenceLength, init = {
      DenseNDArrayFactory.zeros(Shape(this.parser.model.contextVectorsSize))
    })

    this.distancePredictor.propagateErrors(pairsDistanceErrors, optimizer = this.optimizer, copy = false).let {
      contextVectorsErrors.assignSum(it)
    }

    this.depthPredictor.propagateErrors(depthErrors, optimizer = this.optimizer, copy = false).let {
      contextVectorsErrors.assignSum(it)
    }

    this.contextEncoder.propagateErrors(contextVectorsErrors, optimizer = this.optimizer, copy = false).let {
      tokensEncodingsErrors.assignSum(it)
    }

    this.tokensEncoder.propagateErrors(tokensEncodingsErrors, optimizer = this.optimizer)
  }
}
