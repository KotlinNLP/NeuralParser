/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.parsers.birnn

import com.kotlinnlp.neuralparser.parsers.ScorerNetworkConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerConfiguration
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork

/**
 * The builder of NeuralNetworks for the ActionsScorer.
 */
object ActionsScorerNetworkBuilder {

  /**
   * @param inputSize the input layer size
   * @param inputType the input array type
   * @param outputSize the output layer size
   * @param scorerNetworkConfig a scorer network configuration
   *
   * @return a [NeuralNetwork] with the given parameters
   */
  operator fun invoke(inputSize: Int,
                      inputType: LayerType.Input,
                      outputSize: Int,
                      scorerNetworkConfig: ScorerNetworkConfiguration) = NeuralNetwork(

    LayerConfiguration(
      size = inputSize,
      inputType = inputType,
      dropout = scorerNetworkConfig.inputDropout
    ),
    LayerConfiguration(
      size = scorerNetworkConfig.hiddenSize,
      activationFunction = scorerNetworkConfig.hiddenActivation,
      connectionType = LayerType.Connection.Feedforward,
      dropout = scorerNetworkConfig.hiddenDropout,
      meProp = scorerNetworkConfig.hiddenMeProp
    ),
    LayerConfiguration(
      size = outputSize,
      activationFunction = scorerNetworkConfig.outputActivation,
      connectionType = LayerType.Connection.Feedforward,
      meProp = scorerNetworkConfig.outputMeProp
    )
  )
}
