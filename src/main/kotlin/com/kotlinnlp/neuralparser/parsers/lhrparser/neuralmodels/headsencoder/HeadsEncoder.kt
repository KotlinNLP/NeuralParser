/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder

import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.NeuralModel
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder

/**
 * Encoder that generates the Latent Heads Representation.
 *
 * @param model the model of this encoder
 */
class HeadsEncoder(private val model: HeadsEncoderModel) : NeuralModel<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  HeadsEncoderParams // ParamsType
  > {

  /**
   * A BiRNN Encoder that encodes the latent heads.
   */
  private val encoder = BiRNNEncoder<DenseNDArray>(this.model.biRNN)

  /**
   * @param input the vectors that represent each token
   *
   * @return the latent heads representation
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> =
    this.encoder.encode(input, useDropout = true)

  /**
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: List<DenseNDArray>) {
    return this.encoder.backward(errors, propagateToInput = true)
  }

  /**
   * @param copy whether to return by value or by reference
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> = this.encoder.getInputSequenceErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [HeadsEncoder] parameters
   */
  override fun getParamsErrors(copy: Boolean) = HeadsEncoderParams(
    biRNNParameters = this.encoder.getParamsErrors(copy = copy)
  )
}
