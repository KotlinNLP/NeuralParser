/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder

/**
 * Encoder that represents the tokens in their sentential context.
 *
 * @param model the model of this encoder
 * @property useDropout whether to apply the dropout during the forward
 * @property id an identification number useful to track a specific encoder
 */
class ContextEncoder(
  val model: ContextEncoderModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  ContextEncoderParams // ParamsType
  > {

  /**
   * This encoder propagate the errors to the input.
   */
  override val propagateToInput: Boolean = true

  /**
   * The BiRNN Encoder that encodes the tokens into the context vectors.
   */
  private val encoder = DeepBiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN,
    propagateToInput = true,
    useDropout = this.useDropout)

  /**
   * @param input tokens encodings
   *
   * @return the context vectors of the tokens
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> =
    this.encoder.forward(input)

  /**
   * The Backward.
   *
   * @param outputErrors the errors of the current encoding
   */
  override fun backward(outputErrors: List<DenseNDArray>) = this.encoder.backward(outputErrors)

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> = this.encoder.getInputErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [ContextEncoder] parameters
   */
  override fun getParamsErrors(copy: Boolean) = ContextEncoderParams(this.encoder.getParamsErrors(copy = copy))
}
