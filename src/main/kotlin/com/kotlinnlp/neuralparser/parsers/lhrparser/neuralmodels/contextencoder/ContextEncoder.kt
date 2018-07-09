/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder

import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.NeuralModel
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder

/**
 * Encoder that represents the tokens in their sentential context.
 *
 * @param model the model of this encoder
 */
class ContextEncoder(private val model: ContextEncoderModel) : NeuralModel<
  List<DenseNDArray>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray>, // InputErrorsType
  ContextEncoderParams // ParamsType
  > {

  /**
   * The BiRNN Encoder that encodes the tokens into the context vectors.
   */
  private val encoder = DeepBiRNNEncoder<DenseNDArray>(this.model.biRNN)

  /**
   * @param input tokens encodings
   *
   * @return the context vectors of the tokens
   */
  override fun forward(input: List<DenseNDArray>): List<DenseNDArray> =
    this.encoder.encode(sequence = input, useDropout = true)

  /**
   * @param errors the errors of the current encoding
   */
  override fun backward(errors: List<DenseNDArray>) {

    this.encoder.backward(errors, propagateToInput = true)
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
   * @return the errors of the [ContextEncoder] parameters
   */
  override fun getParamsErrors(copy: Boolean) = ContextEncoderParams(this.encoder.getParamsErrors(copy = copy))
}
