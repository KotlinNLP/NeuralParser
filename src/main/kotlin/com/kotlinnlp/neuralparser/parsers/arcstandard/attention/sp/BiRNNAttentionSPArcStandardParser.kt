/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.attention.sp

import com.kotlinnlp.neuralparser.parsers.arcstandard.attention.ArcStandardAttentionFeaturesExtractor
import com.kotlinnlp.neuralparser.parsers.arcstandard.simple.ArcStandardActionsScorer
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos.BiRNNAmbiguousPOSParser
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.AttentionSPStructureFactory
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.AttentionSPSupportStructure
import com.kotlinnlp.neuralparser.utils.features.DenseFeaturesErrors
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import com.kotlinnlp.syntaxdecoder.transitionsystem.ActionsGenerator
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandard
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.GreedyDecoder
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorer
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.BestActionSelector
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.HighestScoreActionSelector
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.MultiActionsSelector
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.MultliActionsSelectorByScore
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractor
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.TransitionSystem
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState

/**
 * A NeuralParser based on the ArcStandard transition system.
 * It uses Embeddings and ambiguous POS vectors to encode the tokens of a sentence through a BiRNN.
 * The features for the actions scoring are composed by the encoding of the token and the embedding vectors of the last
 * applied action through a recurrent system.
 * Actions are scored with a single-prediction Feedforward network with Softmax activation.
 *
 * If the beamSize is 1 then a [GreedyDecoder] is used, a [BeamDecoder] otherwise.
 *
 * @property model the parser model
 * @param wordDropoutCoefficient the word embeddings dropout coefficient (default = 0.0)
 * @param beamSize the max size of the beam (default = 1)
 * @param maxParallelThreads the max number of threads that can run in parallel (default = 1, ignored if beamSize is 1)
 */
class BiRNNAttentionSPArcStandardParser(
  model: BiRNNAttentionSPArcStandardParserModel,
  wordDropoutCoefficient: Double = 0.0,
  beamSize: Int = 1,
  maxParallelThreads: Int = 1
) :
  BiRNNAmbiguousPOSParser<
    StackBufferState,
    ArcStandardTransition,
    DenseFeaturesErrors,
    DenseFeatures,
    AttentionSPSupportStructure,
    BiRNNAttentionSPArcStandardParserModel>
  (
    model = model,
    wordDropoutCoefficient = wordDropoutCoefficient,
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads
  )
{

  /**
   * Whether the action scorer uses the Softmax as output activation (all its networks have the same activation).
   */
  private val useSoftmaxOutput: Boolean =
    this.model.actionsNetwork.layersConfiguration.last().activationFunction == Softmax()

  /**
   * @return the [TransitionSystem] used in this parser
   */
  override fun buildTransitionSystem() = ArcStandard()

  /**
   * @return the [ActionsGenerator] used in this parser
   */
  override fun buildActionsGenerator() =
    ActionsGenerator.Labeled<StackBufferState, ArcStandardTransition>(
      deprels = this.model.corpusDictionary.deprelTags.getElementsReversedSet().groupBy { it.direction })

  /**
   * @return the [ActionsScorer] used in this parser
   */
  override fun buildActionsScorer() = ArcStandardActionsScorer<AttentionSPSupportStructure, TokensAmbiguousPOSContext>(
    network = this.model.actionsNetwork,
    optimizer = ParamsOptimizer(
      params = this.model.actionsNetwork.model,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)),
    deprelTags = this.model.corpusDictionary.deprelTags)

  /**
   * @return the [SupportStructureFactory] used in this parser
   */
  override fun buildSupportStructureFactory() = AttentionSPStructureFactory(
    actionMemoryRNN = this.model.actionMemoryRNN,
    transformLayerParams = this.model.transformLayerParams,
    stateAttentionNetworkParams = this.model.stateAttentionNetworkParams,
    featuresLayerParams = this.model.featuresLayerParams,
    actionsNetwork = this.model.actionsNetwork,
    outputErrorsInit = if (this.useSoftmaxOutput){
      OutputErrorsInit.AllErrors
    } else {
      OutputErrorsInit.AllZeros
    }
  )

  /**
   * @return the [FeaturesExtractor] used in this parser
   */
  override fun buildFeaturesExtractor() = ArcStandardAttentionFeaturesExtractor<AttentionSPSupportStructure>(
    actionsVectors = this.model.actionsVectors,
    actionsVectorsOptimizer = ActionsVectorsOptimizer(
      actionsVectorsMap = this.model.actionsVectors,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)),
    transformLayerOptimizer = ParamsOptimizer(
      params = this.model.transformLayerParams,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)),
    memoryRNNOptimizer = ParamsOptimizer(
      params = this.model.actionMemoryRNN.model,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)),
    stateAttentionNetworkOptimizer = ParamsOptimizer(
      params = this.model.stateAttentionNetworkParams,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)),
    featuresLayerOptimizer = ParamsOptimizer(
      params = this.model.featuresLayerParams,
      updateMethod = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999)),
    memoryEncodingSize = this.model.actionMemoryRNN.layersConfiguration.last().size,
    featuresSize = this.model.featuresSize,
    posTags = this.model.corpusDictionary.posTags,
    deprelTags = this.model.corpusDictionary.deprelTags)

  /**
   * @return the [BestActionSelector] used in this parser during greedy decoding
   */
  override fun buildBestActionSelector() = HighestScoreActionSelector<
    StackBufferState,
    ArcStandardTransition,
    DenseItem,
    TokensAmbiguousPOSContext>()

  /**
   * @return the [MultiActionsSelector] used in this parser during beam decoding
   */
  override fun buildMultiActionsSelector() = MultliActionsSelectorByScore<
    StackBufferState,
    ArcStandardTransition,
    DenseItem,
    TokensAmbiguousPOSContext>()

  /**
   * Callback called before applying an action.
   */
  override fun beforeApplyAction(action: Transition<ArcStandardTransition, StackBufferState>.Action,
                                 context: TokensAmbiguousPOSContext) = Unit
}
