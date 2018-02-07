/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased

import com.kotlinnlp.neuralparser.parsers.transitionbased.models.ScorerNetworkConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionNetworkConfig
import com.kotlinnlp.simplednn.utils.Serializer
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The serializable model of a [NeuralParser].
 *
 * @property scoreAccumulatorFactory a factory of score accumulators
 */
abstract class NeuralParserModel(
  val scoreAccumulatorFactory: ScoreAccumulator.Factory
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [NeuralParserModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [NeuralParserModel]
     *
     * @return the [NeuralParserModel] read from [inputStream] and decoded
     */
    fun <ModelType: NeuralParserModel> load(inputStream: InputStream): ModelType = Serializer.deserialize(inputStream)
  }

  /**
   * Serialize this [NeuralParserModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [NeuralParserModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * Utility method: build the configuration of a multi-prediction network with dense input.
   *
   * @param inputSize the input size
   * @param outputSize the output size
   * @param scorerNetworksConfig the scorer network configuration object
   *
   * @return the configuration of a multi-prediction network
   */
  protected fun buildMPNetworkConfig(inputSize: Int,
                                     outputSize: Int,
                                     scorerNetworksConfig: ScorerNetworkConfiguration) = MultiPredictionNetworkConfig(
    inputSize = inputSize,
    inputType = LayerType.Input.Dense,
    inputDropout = scorerNetworksConfig.inputDropout,
    hiddenSize = scorerNetworksConfig.hiddenSize,
    hiddenActivation = scorerNetworksConfig.hiddenActivation,
    hiddenDropout = scorerNetworksConfig.hiddenDropout,
    hiddenMeProp = scorerNetworksConfig.hiddenMeProp,
    outputSize = outputSize,
    outputActivation = scorerNetworksConfig.outputActivation,
    outputMeProp = scorerNetworksConfig.outputMeProp
  )
}
