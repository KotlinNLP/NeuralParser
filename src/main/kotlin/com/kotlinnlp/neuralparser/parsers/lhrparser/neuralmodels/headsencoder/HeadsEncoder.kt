/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder

/**
 * Encoder that generates the Latent Heads Representation.
 *
 * @param model the model of this encoder
 * @property useDropout whether to apply the dropout during the forward
 * @property id an identification number useful to track a specific encoder
 */
class HeadsEncoder(
  val model: HeadsEncoderModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  HeadsEncoderParams // ParamsType
  > {

  /**
   * This encoder propagate the errors to the input.
   */
  override val propagateToInput: Boolean = true

  /**
   * A BiRNN Encoder that encodes the latent heads.
   */
  private val encoder = BiRNNEncoder<DenseNDArray>(
    network = this.model.biRNN,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * @param input the vectors that represent each token
   *
   * @return the latent heads representation
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> = this.encoder.forward(input)

  /**
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
   * @return the errors of the [HeadsEncoder] parameters
   */
  override fun getParamsErrors(copy: Boolean) = HeadsEncoderParams(
    biRNNParameters = this.encoder.getParamsErrors(copy = copy)
  )
}
