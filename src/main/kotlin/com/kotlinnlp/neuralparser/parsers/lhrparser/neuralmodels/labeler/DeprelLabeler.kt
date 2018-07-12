/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler

import com.kotlinnlp.neuralparser.parsers.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.Deprel
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
  List<DeprelLabeler.Prediction>, // OutputType
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
   * The outcome of a single prediction of the labeler.
   *
   * @property deprels the deprels prediction
   */
  data class Prediction(val deprels: DenseNDArray)

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
   * Predict the deprel and the POS tag for each token.
   *
   * @param input the input
   *
   * @return a list of predictions, one for each token
   */
  override fun forward(input: Input): List<Prediction> {

    this.dependencyTree = input.dependencyTree

    val outputList: List<DenseNDArray> = this.processor.forward(ArrayList(this.extractFeatures(input)))

    return outputList.map { Prediction(deprels = it) }
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

    inputErrors.forEachIndexed { tokenId, (depErrors, govErrors) ->

      val depVector: DenseNDArray = contextErrors[tokenId]
      val govVector: DenseNDArray = this.dependencyTree.heads[tokenId].let {
        if (it == null) rootErrors else contextErrors[it]
      }

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
   * @return the [Deprel] corrisponding to the given [index]
   */
  fun getDeprel(index: Int) = this.model.deprels.getElement(index)!!

  /**
   * @return a list of features
   */
  private fun extractFeatures(input: Input): List<List<DenseNDArray>> {

    val features = mutableListOf<List<DenseNDArray>>()

    input.lss.sentence.tokens.map { it.id }.zip(this.dependencyTree.heads).forEach { (dependentId, headId) ->

      features.add(listOf(
        input.lss.contextVectors[dependentId],
        headId?.let { input.lss.contextVectors[it] } ?: input.lss.virtualRoot
      ))
    }

    return features
  }
}
