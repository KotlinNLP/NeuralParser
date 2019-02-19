/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.lssencoder.tokensencoder.LSSTokensEncoderModel
import com.kotlinnlp.morphodisambiguator.language.MorphoDictionary
import com.kotlinnlp.morphodisambiguator.language.PropertyNames
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.ConcatFeedforwardMerge
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNConfig
import com.kotlinnlp.simplednn.deeplearning.sequenceencoder.ParallelEncoderModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

class MorphoDisambiguatorModel (
    val corpusMorphologies: MorphoDictionary,
    val lssModel: LSSTokensEncoderModel<ParsingToken, ParsingSentence>,
    val biRNNConfig: BiRNNConfig,
    val config: MorphoDisambiguatorConfiguration
): Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [MorphoDisambiguatorModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [MorphoDisambiguatorModel]
     *
     * @return the [MorphoDisambiguatorModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): MorphoDisambiguatorModel = Serializer.deserialize(inputStream)
  }

  /**
   * Serialize this [MorphoDisambiguatorModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [MorphoDisambiguatorModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)

  /**
   * The model of the BiRNN encoder.
   */
  val biRNNEncoderModel = BiRNN(
      inputType = LayerType.Input.Dense,
      inputSize = this.lssModel.tokenEncodingSize,
      dropout = 0.0,
      recurrentConnectionType = biRNNConfig.connectionType,
      hiddenActivation = biRNNConfig.hiddenActivation,
      hiddenSize = config.BiRNNHiddenSize,
      outputMergeConfiguration = ConcatFeedforwardMerge(outputSize = config.BIRNNOutputSize))

  /**
   * The list of seed to initialize parameters
   */

  private val randomSeedValues: List<Long> = listOf(617, 499, 383, 181, 887, 113, 911)

  /**
   * The models of the feedforward morphology classifiers
   */
  private val feedforwardMorphoClassifiers: List<NeuralNetwork> = PropertyNames().propertyNames.mapIndexed {
    i, name ->
    NeuralNetwork(LayerInterface(size = config.BIRNNOutputSize),
        LayerInterface(
            size = config.parallelEncodersHiddenSize,
            connectionType = LayerType.Connection.Feedforward,
            activationFunction = config.parallelEncodersActivationFunction),
        LayerInterface(
            type = LayerType.Input.Dense,
            size = corpusMorphologies.morphologyDictionaryMap[name]!!.size,
            dropout = 0.0,
            connectionType = LayerType.Connection.Feedforward,
            activationFunction = Softmax()),
        weightsInitializer = GlorotInitializer(seed = randomSeedValues[i]),
        biasesInitializer = GlorotInitializer(seed = randomSeedValues[i]))
  }

  /**
   * The sequence parallel encoder model
   */
  val sequenceParallelEncoderModel: ParallelEncoderModel = ParallelEncoderModel(
      networks = feedforwardMorphoClassifiers
  )


}