/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.mergeconfig.ConcatFeedforwardMerge
import java.io.Serializable

/**
 * The model of the [HeadsEncoder].
 *
 * @property tokenEncodingSize the size of the token encoding vectors
 * @property connectionType the recurrent connection type of the BiRNN used to encode tokens
 * @property hiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param recurrentDropout the probability of the recurrent dropout (default 0.0)
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class HeadsEncoderModel(
  val tokenEncodingSize: Int,
  val connectionType: LayerType.Connection,
  val hiddenActivation: ActivationFunction?,
  recurrentDropout: Double = 0.0,
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

  /**
   * The BiRNN of the HeadsEncoder.
   */
  val biRNN = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = this.tokenEncodingSize,
    dropout = recurrentDropout,
    recurrentConnectionType = this.connectionType,
    hiddenActivation = this.hiddenActivation,
    hiddenSize = this.tokenEncodingSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer,
    outputMergeConfiguration = ConcatFeedforwardMerge(outputSize = this.tokenEncodingSize))
}