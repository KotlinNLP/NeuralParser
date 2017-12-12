/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction

/**
 * The configuration of a scorer network.
 *
 * @property inputDropout the probability of dropout of the input layer (default = 0.0)
 * @property hiddenSize the hidden layer size
 * @property hiddenActivation the hidden layer activation function
 * @property hiddenDropout the probability of dropout of the hidden layer (default = 0.0)
 * @property hiddenMeProp whether to use the 'meProp' errors propagation algorithm for the hidden layer (default false)
 * @property outputActivation the output layer activation function
 * @property outputMeProp whether to use the 'meProp' errors propagation algorithm for the output layer (default false)
 */
data class ScorerNetworkConfiguration(
  val inputDropout: Double = 0.0,
  val hiddenSize: Int,
  val hiddenActivation: ActivationFunction,
  val hiddenDropout: Double = 0.0,
  val hiddenMeProp: Boolean = false,
  val outputActivation: ActivationFunction?,
  val outputMeProp: Boolean = false
)
