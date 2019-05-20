/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistaceparser

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNN
import com.kotlinnlp.tokensencoder.wrapper.TokensEncoderWrapperModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream

/**
 * The model of the [StructuralDistanceParser].
 *
 * @param language
 * @property tokensEncoderModel the model of the Tokens0 encoder
 */
class StructuralDistanceParserModel(
  language: Language,
  val tokensEncoderModel: TokensEncoderWrapperModel<ParsingToken, ParsingSentence, *, *>,
  contextBiRNNConfig: BiRNNConfig
) : NeuralParserModel(language) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [StructuralDistanceParserModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [StructuralDistanceParserModel]
     *
     * @return the [StructuralDistanceParserModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): StructuralDistanceParserModel = Serializer.deserialize(inputStream)
  }

  /**
   * The size of the encoded tokens.
   */
  val tokenEncodingSize: Int = this.tokensEncoderModel.model.tokenEncodingSize

  /**
   * The model of the context encoder.
   */
  val contextEncoderModel = if (contextBiRNNConfig.numberOfLayers == 2)
    DeepBiRNN(
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokenEncodingSize,
        dropout = 0.0,
        recurrentConnectionType = contextBiRNNConfig.connectionType,
        hiddenSize = this.tokenEncodingSize,
        hiddenActivation = contextBiRNNConfig.hiddenActivation,
        biasesInitializer = null),
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokenEncodingSize * 2,
        dropout = 0.0,
        recurrentConnectionType = contextBiRNNConfig.connectionType,
        hiddenSize = this.tokenEncodingSize,
        hiddenActivation = contextBiRNNConfig.hiddenActivation,
        biasesInitializer = null))
  else
    DeepBiRNN(
      BiRNN(
        inputType = LayerType.Input.Dense,
        inputSize = this.tokenEncodingSize,
        dropout = 0.0,
        recurrentConnectionType = contextBiRNNConfig.connectionType,
        hiddenSize = this.tokenEncodingSize,
        hiddenActivation = contextBiRNNConfig.hiddenActivation,
        biasesInitializer = null))

  /**
   * The size of the context vectors.
   */
  val contextVectorsSize: Int = this.contextEncoderModel.outputSize

  /**
   * The parameters of the model that predicts the structural distance among pairs of nodes.
   */
  val distanceModel = StackedLayersParameters(
    LayerInterface(sizes = listOf(this.contextVectorsSize, this.contextVectorsSize)),
    LayerInterface(size = this.contextVectorsSize, connectionType = LayerType.Connection.Sub),
    LayerInterface(size = 1, connectionType = LayerType.Connection.SquaredDistance))

  /**
   * The parameters of the model that predicts the structural depth of a node.
   */
  val depthModel = StackedLayersParameters(
    LayerInterface(size = this.contextVectorsSize),
    LayerInterface(size = 1, connectionType = LayerType.Connection.SquaredDistance))
}
