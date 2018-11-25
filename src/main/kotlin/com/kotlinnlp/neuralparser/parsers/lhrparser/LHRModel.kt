/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.lssencoder.LSSModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.LabelerModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.LossCriterionType
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.morphopercolator.MorphoPercolator
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.selector.LabelerSelector
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.selector.NoFilterSelector
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.AffineMerge
import com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork.PointerNetworkModel
import com.kotlinnlp.utils.Serializer
import java.io.InputStream

/**
 * The model of the [LHRParser].
 *
 * @property language the language within the parser works (default = unknown)
 * @param corpusDictionary a corpus dictionary
 * @property lssModel the model of the LSS encoder
 * @property useLabeler whether to use the labeler
 * @property lossCriterionType the training mode of the labeler
 * @property predictPosTags whether to predict the POS tags together with the Deprels
 * @property labelerSelector the labeler prediction selector (no filter by default)
 * @property morphoPercolator a percolator of morphology properties
 */
class LHRModel(
  corpusDictionary: CorpusDictionary,
  val lssModel: LSSModel<ParsingToken, ParsingSentence>,
  val useLabeler: Boolean,
  val lossCriterionType: LossCriterionType,
  val predictPosTags: Boolean,
  val labelerSelector: LabelerSelector = NoFilterSelector,
  val morphoPercolator: MorphoPercolator
) : NeuralParserModel(lssModel.language) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [LHRModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [LHRModel]
     *
     * @return the [LHRModel] read from [inputStream] and decoded
     */
    fun load(inputStream: InputStream): LHRModel = Serializer.deserialize(inputStream)
  }

  /**
   * The score threshold above which to consider a labeler output valid.
   */
  val labelerScoreThreshold = 1.0 / corpusDictionary.grammaticalConfigurations.size

  /**
   * The model of the Labeler.
   */
  val labelerModel: LabelerModel? = if (this.useLabeler)
    LabelerModel(
      contextEncodingSize = this.lssModel.contextVectorsSize,
      grammaticalConfigurations = corpusDictionary.grammaticalConfigurations,
      lossCriterionType = this.lossCriterionType)
  else
    null

  /**
   * The model of the pointer network used for the positional encoding.
   */
  val pointerNetworkModel = PointerNetworkModel(
    inputSize = this.lssModel.contextVectorsSize,
    vectorSize = this.lssModel.contextVectorsSize,
    mergeConfig = AffineMerge(
      outputSize = 100,
      activationFunction = Tanh()))

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    %-33s : %s
    %-33s : %s
    %-33s : %s
    %-33s : %s
    %-33s : %s
  """.trimIndent().format(
    this.lssModel.tokensEncoderWrapperModel.model::class.simpleName, this.lssModel.tokensEncoderWrapperModel.model,
    "Context Encoder", this.lssModel.contextBiRNNConfig,
    "Heads Encoder", this.lssModel.headsBiRNNConfig,
    "Labeler training mode", this.lossCriterionType,
    "Predict POS tags", this.predictPosTags
  )
}
