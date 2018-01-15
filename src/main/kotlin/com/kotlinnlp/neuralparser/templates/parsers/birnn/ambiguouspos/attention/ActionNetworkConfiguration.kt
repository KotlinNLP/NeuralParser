/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos.attention

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType

/**
 * The configuration of the Recurrent Neural Network used to decode the actions features.
 *
 * @property connectionType the type of recurrent connection
 * @property activation the activation function
 * @property outputSize the output layer size
 * @property dropout the probability of dropout (default 0.0)
 * @property meProp whether to use the 'meProp' errors propagation algorithm for the output layer (default false)
 */
data class ActionNetworkConfiguration(
  val connectionType: LayerType.Connection,
  val activation: ActivationFunction,
  val outputSize: Int,
  val dropout: Double = 0.0,
  val meProp: Boolean = false) {

  /**
   * Check the connection type.
   */
  init {
    require(this.connectionType.property == LayerType.Property.Recurrent) {
      "The hidden connection type of the network that decodes the action features must be Recurrent."
    }
  }
}
