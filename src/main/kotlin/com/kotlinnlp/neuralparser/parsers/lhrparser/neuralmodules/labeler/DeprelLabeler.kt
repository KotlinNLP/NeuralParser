/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler

import com.kotlinnlp.neuralparser.parsers.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprel
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredDeprelList
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.sortByScore
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The Labeler.
 *
 * @param model the model of this labeler
 * @property useDropout whether to apply the dropout during the forward
 * @property id an identification number useful to track a specific encoder
 */
class DeprelLabeler(
  private val model: DeprelLabelerModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  DeprelLabeler.Input, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  DeprelLabeler.InputErrors, // InputErrorsType
  DeprelLabelerParams // ParamsType
  > {

  /**
   * The input of this labeler.
   *
   * @param lss the latent syntactic structure
   * @param dependencyTree the dependency tree
   */
  class Input(val lss: LatentSyntacticStructure, val dependencyTree: DependencyTree)

  /**
   * This encoder propagate the errors to the input.
   */
  override val propagateToInput: Boolean = true

  /**
   * The input errors of this labeler.
   *
   * @param rootErrors the errors of the virtual root
   * @param contextErrors the errors of the context vectors
   */
  class InputErrors(
    val rootErrors: DenseNDArray,
    val contextErrors: List<DenseNDArray>)

  /**
   * The processor that classify the deprels.
   */
  private val processor = BatchFeedforwardProcessor<DenseNDArray>(
    neuralNetwork = this.model.networkModel,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The dependency tree used for the last predictions done.
   */
  private lateinit var dependencyTree: DependencyTree

  /**
   * Predict the possible deprels for each token.
   *
   * @param input the input
   *
   * @return a list of deprels for each token sorted by descending order
   */
  fun predict(input: Input): List<ScoredDeprelList> = this.forward(input).map { prediction ->
    (0 until prediction.length).map { i -> ScoredDeprel(this.getDeprel(i), score = prediction[i]) }.sortByScore()
  }

  /**
   * Return the network outcomes for each token.
   *
   * @param input the input
   *
   * @return the network outcomes for each token
   */
  override fun forward(input: Input): List<DenseNDArray> {

    this.dependencyTree = input.dependencyTree

    return this.processor.forward(ArrayList(this.extractFeatures(input.lss)))
  }

  /**
   * Propagate the errors through the neural components of the labeler.
   *
   * @param outputErrors the list of errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.processor.backward(outputErrors)
  }

  /**
   * @return the input errors and the root errors
   */
  override fun getInputErrors(copy: Boolean): InputErrors {

    val inputErrors: List<List<DenseNDArray>> = this.processor.getInputsErrors(copy = false)

    val contextErrors = List(size = inputErrors.size, init = {
      DenseNDArrayFactory.zeros(Shape(this.model.contextEncodingSize))
    })

    val rootErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.contextEncodingSize))

    inputErrors.forEachIndexed { tokenIndex, (depErrors, govErrors) ->

      val tokenId: Int = this.dependencyTree.elements[tokenIndex]
      val depVector: DenseNDArray = contextErrors[tokenIndex]
      val govVector: DenseNDArray = this.dependencyTree.getHead(tokenId)?.let {
        contextErrors[this.dependencyTree.getPosition(it)]
      } ?: rootErrors

      depVector.assignSum(depErrors)
      govVector.assignSum(govErrors)
    }

    return InputErrors(rootErrors = rootErrors, contextErrors = contextErrors)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [DeprelLabeler] parameters
   */
  override fun getParamsErrors(copy: Boolean) = DeprelLabelerParams(this.processor.getParamsErrors(copy = copy))

  /**
   * @param index a prediction index
   *
   * @return the [Deprel] with the given [index]
   */
  private fun getDeprel(index: Int) = this.model.deprels.getElement(index)!!

  /**
   * @param lss the latent syntactic structure of the input sentence
   *
   * @return the list of features that encode the sentence tokens
   */
  private fun extractFeatures(lss: LatentSyntacticStructure): List<List<DenseNDArray>> =
    lss.sentence.tokens.map { token ->
      listOf(
        lss.getContextVectorById(token.id),
        this.dependencyTree.getHead(token.id)?.let { lss.getContextVectorById(it) } ?: lss.virtualRoot
      )
    }
}
