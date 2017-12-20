/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.charbased

import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ActionsScorerNetworkBuilder
import com.kotlinnlp.neuralparser.templates.parsers.birnn.charbased.CharBasedBiRNNParserModel
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * The parser model for the ArcStandard parser based on the birnn
 *
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property charEmbeddingSize the size of each char embedding vector
 * @property charsEncodingSize the size of each char encoding vector (the encoding of more char embeddings)
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property hanAttentionSize the size of the attention arrays
 * @property hanConnectionType the recurrent connection type of the HAN
 * @property hanHiddenActivation the hidden activation function of the HAN BiRNN networks
 * @property biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @property biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param scorerNetworkConfig the configuration of the scorer network
 */
class CharBasedBiRNNArcStandardParserModel(
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  corpusDictionary: CorpusDictionary,
  charEmbeddingSize: Int,
  charsEncodingSize: Int,
  wordEmbeddingSize: Int,
  hanAttentionSize: Int,
  hanConnectionType: LayerType.Connection,
  hanHiddenActivation: ActivationFunction,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction,
  scorerNetworkConfig: ScorerNetworkConfiguration
) :
  CharBasedBiRNNParserModel(
  scoreAccumulatorFactory = scoreAccumulatorFactory,
    corpusDictionary = corpusDictionary,
    charEmbeddingSize = charEmbeddingSize,
    charsEncodingSize = charsEncodingSize,
    wordEmbeddingSize = wordEmbeddingSize,
    hanAttentionSize = hanAttentionSize,
    hanConnectionType = hanConnectionType,
    hanHiddenActivation = hanHiddenActivation,
    biRNNConnectionType = biRNNConnectionType,
    biRNNHiddenActivation = biRNNHiddenActivation) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable)
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The neural network of the actions scorer.
   */
  val actionsScorerNetwork: NeuralNetwork = ActionsScorerNetworkBuilder(
    inputSize = 4 * this.biRNN.outputSize, // the input is a tokens window of 4 elements
    inputType = LayerType.Input.Dense,
    outputSize = this.corpusDictionary.deprelTags.size + 1, // Deprels + Shift
    scorerNetworkConfig = scorerNetworkConfig
  )
}
