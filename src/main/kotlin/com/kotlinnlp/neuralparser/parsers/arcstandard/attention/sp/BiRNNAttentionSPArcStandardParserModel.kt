/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.attention.sp

import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.language.CorpusDictionary
import com.kotlinnlp.neuralparser.parsers.ScorerNetworkConfiguration
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos.attention.BiRNNAttentionSPParserModel
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos.attention.MemoryRNNConfiguration
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator

/**
 * The parser model for the ArcStandard parser based on the BiRNN with single-prediction scoring with attention
 * features encoding.
 *
 * @property actionsScoresActivation the function used to activate the actions scores (can be null)
 * @property scoreAccumulatorFactory a factory of score accumulators
 * @property corpusDictionary a corpus dictionary
 * @property nounDefaultPOSTag the 'Noun' POS tag used for title case unknown forms
 * @property otherDefaultPOSTags the list of POS tags used for other unknown forms
 * @property wordEmbeddingSize the size of each word embedding vector
 * @property posEmbeddingSize the size of each POS embedding vector
 * @property actionsEmbeddingsSize the size of each action embedding vector
 * @property attentionSize the attention size of the state attention network
 * @property featuresSize the features size
 * @property preTrainedWordEmbeddings pre-trained word embeddings to add to the tokens encodings (default = null)
 * @param biRNNConnectionType the recurrent connection type of the BiRNN used to encode tokens
 * @param biRNNHiddenActivation the hidden activation function of the BiRNN used to encode tokens
 * @param biRNNLayers number of stacked BiRNNs
 * @param memoryRNNConfig the configuration of the memory RNN
 * @param scorerNetworkConfig the configuration of the scorer network
 */
class BiRNNAttentionSPArcStandardParserModel(
  actionsScoresActivation: ActivationFunction?,
  scoreAccumulatorFactory: ScoreAccumulator.Factory,
  corpusDictionary: CorpusDictionary,
  nounDefaultPOSTag: POSTag,
  otherDefaultPOSTags: List<POSTag>,
  wordEmbeddingSize: Int,
  posEmbeddingSize: Int,
  actionsEmbeddingsSize: Int,
  attentionSize: Int,
  featuresSize: Int,
  preTrainedWordEmbeddings: EmbeddingsMap<String>? = null,
  biRNNConnectionType: LayerType.Connection,
  biRNNHiddenActivation: ActivationFunction?,
  biRNNLayers: Int,
  memoryRNNConfig: MemoryRNNConfiguration,
  scorerNetworkConfig: ScorerNetworkConfiguration
) : BiRNNAttentionSPParserModel(
  actionsScoresActivation = actionsScoresActivation,
  scoreAccumulatorFactory = scoreAccumulatorFactory,
  corpusDictionary = corpusDictionary,
  nounDefaultPOSTag = nounDefaultPOSTag,
  otherDefaultPOSTags = otherDefaultPOSTags,
  wordEmbeddingSize = wordEmbeddingSize,
  posEmbeddingSize = posEmbeddingSize,
  actionsEmbeddingsSize = actionsEmbeddingsSize,
  attentionSize = attentionSize,
  featuresSize = featuresSize,
  preTrainedWordEmbeddings = preTrainedWordEmbeddings,
  biRNNConnectionType = biRNNConnectionType,
  biRNNHiddenActivation = biRNNHiddenActivation,
  biRNNLayers = biRNNLayers,
  memoryRNNConfig = memoryRNNConfig,
  scorerNetworkConfig = scorerNetworkConfig,
  numberOfTransitionsVectors = 4 // Shift and Wait considered the same
) {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}

