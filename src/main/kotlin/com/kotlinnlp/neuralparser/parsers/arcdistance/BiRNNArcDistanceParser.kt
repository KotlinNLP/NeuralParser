/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcdistance

import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.templates.parsers.birnn.simple.BiRNNParser
import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.neuralparser.templates.supportstructure.singleprediction.SPSupportStructure
import com.kotlinnlp.neuralparser.templates.supportstructure.singleprediction.SPStructureFactory
import com.kotlinnlp.neuralparser.utils.features.DenseFeaturesErrors
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import com.kotlinnlp.syntaxdecoder.transitionsystem.ActionsGenerator
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
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcdistance.ArcDistance
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcdistance.ArcDistanceTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.scoreaccumulator.ScoreAccumulator
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState

/**
 * A NeuralParser based on the ArcDistance transition system that uses Embeddings encoded with a BiRNN to encode the
 * tokens of a sentence.
 *
 * If the beamSize is 1 then a [GreedyDecoder] is used, a [BeamDecoder] otherwise.
 *
 * @property model the parser model
 * @param wordDropoutCoefficient the word embeddings dropout coefficient (default = 0.0)
 * @param posDropoutCoefficient the POS embeddings dropout coefficient (default = 0.0)
 * @param beamSize the max size of the beam (default = 1)
 * @param maxParallelThreads the max number of threads that can run in parallel (default = 1, ignored if beamSize is 1)
 */
class BiRNNArcDistanceParser(
  model: BiRNNArcDistanceParserModel,
  wordDropoutCoefficient: Double = 0.0,
  posDropoutCoefficient: Double = 0.0,
  beamSize: Int = 1,
  maxParallelThreads: Int = 1
) :
  BiRNNParser<
    StackBufferState,
    ArcDistanceTransition,
    DenseFeaturesErrors,
    DenseFeatures,
    SPSupportStructure,
    BiRNNArcDistanceParserModel>
  (
    model = model,
    wordDropoutCoefficient = wordDropoutCoefficient,
    posDropoutCoefficient = posDropoutCoefficient,
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads
  )
{

  /**
   * @return the [TransitionSystem] used in this parser
   */
  override fun buildTransitionSystem() = ArcDistance()

  /**
   * @return the [ActionsGenerator] used in this parser
   */
  override fun buildActionsGenerator() = ActionsGenerator.Labeled<StackBufferState, ArcDistanceTransition>(
    deprels = this.model.corpusDictionary.deprelTags.getElementsReversedSet().groupBy { it.direction })

  /**
   * @return the [ActionsScorer] used in this parser
   */
  override fun buildActionsScorer() = ArcDistanceActionsScorer(
    network = this.model.actionsScorerNetwork,
    optimizer = ParamsOptimizer(
      params = this.model.actionsScorerNetwork.model,
      updateMethod = ADAMMethod(stepSize = 0.001)),
    deprelTags = this.model.corpusDictionary.deprelTags)

  /**
   * @return the [SupportStructureFactory] used in this parser
   */
  override fun buildSupportStructureFactory() = SPStructureFactory(
    network = this.model.actionsScorerNetwork,
    outputErrorsInit = OutputErrorsInit.AllZeros)

  /**
   * @return the [FeaturesExtractor] used in this parser
   */
  override fun buildFeaturesExtractor() = ArcDistanceEmbeddingsFeaturesExtractor()

  /**
   * @return the [BestActionSelector] used in this parser during greedy decoding
   */
  override fun buildBestActionSelector() = HighestScoreActionSelector<
    StackBufferState,
    ArcDistanceTransition,
    DenseItem,
    TokensEmbeddingsContext>()

  /**
   * @return the [MultiActionsSelector] used in this parser during beam decoding
   */
  override fun buildMultiActionsSelector() = MultliActionsSelectorByScore<
    StackBufferState,
    ArcDistanceTransition,
    DenseItem,
    TokensEmbeddingsContext>()

  /**
   * @return the [ScoreAccumulator] factory used in this parser
   */
  override fun buildScorerAccumulatorFactory() = this.model.scoreAccumulatorFactory

  /**
   * Callback called before applying an action.
   */
  override fun beforeApplyAction(action: Transition<ArcDistanceTransition, StackBufferState>.Action,
                                 context: TokensEmbeddingsContext) = Unit
}
