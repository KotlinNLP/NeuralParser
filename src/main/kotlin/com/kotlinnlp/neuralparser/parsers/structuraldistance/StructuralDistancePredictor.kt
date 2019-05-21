/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistance

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The StructuralDistancePredictor.
 *
 * @param model the model of this labeler
 * @property useDropout whether to apply the dropout during the forward
 * @property id an identification number useful to track a specific encoder
 */
class StructuralDistancePredictor(
  private val model: StackedLayersParameters,
  override val useDropout: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  StructuralDistancePredictor.Input, // InputType
  List<Double>, // OutputType
  List<Double>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   *
   */
  class Input(val hiddens: List<DenseNDArray>, val pairs: List<Pair<Int, Int>>)

  /**
   * This encoder propagate the errors to the input.
   */
  override val propagateToInput: Boolean = true

  /**
   * The processor that classify the grammar of a token.
   */
  private val processor = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.model,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   *
   */
  private lateinit var lastInput: Input

  /**
   * Return the distance of each pair.
   *
   * @param input all the pairs
   *
   * @return the distance of each pair
   */
  override fun forward(input: Input): List<Double> {

    this.lastInput = input

    return this.processor.forward(ArrayList(
      input.pairs.map { (i, j) -> listOf(input.hiddens[i], input.hiddens[j]) }
    )).map { it.expectScalar() }
  }

  /**
   * The backward.
   *
   * @param outputErrors the list of errors
   */
  override fun backward(outputErrors: List<Double>) {

    this.processor.backward(outputErrors.map { DenseNDArrayFactory.scalarOf(it) })
  }

  /**
   * @return the input errors and the root errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> {

    val pairsErrors: List<List<DenseNDArray>> = this.processor.getInputsErrors(copy = false)

    val hiddenErrors = List(size = this.lastInput.hiddens.size, init = {
      DenseNDArrayFactory.zeros(Shape(this.model.layersConfiguration.first().sizes.first()))
    })

    pairsErrors.forEachIndexed { pairIndex, (iErrors, jErrors) ->

      val (i, j) = this.lastInput.pairs[pairIndex]

      hiddenErrors[i].assignSum(iErrors)
      hiddenErrors[j].assignSum(jErrors)
    }

    return hiddenErrors
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [StructuralDistancePredictor] parameters
   */
  override fun getParamsErrors(copy: Boolean) = this.processor.getParamsErrors(copy = copy)
}
