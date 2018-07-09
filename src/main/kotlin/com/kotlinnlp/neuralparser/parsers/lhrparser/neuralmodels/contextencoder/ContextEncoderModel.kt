/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNN
import java.io.Serializable

/**
 * The model of the [ContextEncoder].
 *
 * @property tokenEncodingSize the size of the token encoding vectors
 * @property hiddenActivation the activation function of the hidden layer
 * @property connectionType type of recurrent neural network (e.g. LSTM, GRU, CFN, SimpleRNN)
 * @property numberOfLayers number of stacked BiRNNs (default 1)
 * @param dropout the probability of the recurrent dropout (default 0.0)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class ContextEncoderModel(
  val tokenEncodingSize: Int,
  val connectionType: LayerType.Connection,
  val hiddenActivation: ActivationFunction?,
  val numberOfLayers: Int = 1,
  dropout: Double,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  init { require(this.numberOfLayers in 1..2) { "Invalid number of layers: ${this.numberOfLayers}"} }

  /**
   * The BiRNN of the ContextEncoder.
   */
  val biRNN = if (this.numberOfLayers == 2) {
    DeepBiRNN(
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokenEncodingSize,
        dropout = dropout,
        recurrentConnectionType = this.connectionType,
        hiddenSize = this.tokenEncodingSize,
        hiddenActivation = this.hiddenActivation,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer),
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokenEncodingSize * 2,
        dropout = dropout,
        recurrentConnectionType = this.connectionType,
        hiddenSize = this.tokenEncodingSize,
        hiddenActivation = this.hiddenActivation,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer))
  } else {
    DeepBiRNN(
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokenEncodingSize,
        dropout = dropout,
        recurrentConnectionType = this.connectionType,
        hiddenSize = this.tokenEncodingSize,
        hiddenActivation = this.hiddenActivation,
        weightsInitializer = weightsInitializer,
        biasesInitializer = biasesInitializer))
  }

  /**
   * The size of the output context vectors.
   */
  val contextEncodingSize: Int = this.biRNN.outputSize
}