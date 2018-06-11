/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine

import com.kotlinnlp.neuralparser.parsers.transitionbased.models.SimpleArcEagerSpineActionsScorer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensEmbeddingsContext
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.parsers.birnn.simple.BiRNNParser
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.multiprediction.MPSupportStructure
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.multiprediction.MPStructureFactory
import com.kotlinnlp.neuralparser.utils.features.GroupedDenseFeatures
import com.kotlinnlp.neuralparser.utils.features.GroupedDenseFeaturesErrors
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionOptimizer
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import com.kotlinnlp.syntaxdecoder.GreedyDecoder
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.HighestScoreActionSelector
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.MultliActionsSelectorByScore
import com.kotlinnlp.syntaxdecoder.transitionsystem.ActionsGenerator
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpine
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineState
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineTransition

/**
 * A NeuralParser based on the ArcEagerSpine transition system that uses Embeddings encoded with a BiRNN to encode the
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
class BiRNNArcEagerSpineParser(
  model: BiRNNArcEagerSpineParserModel,
  wordDropoutCoefficient: Double = 0.0,
  posDropoutCoefficient: Double = 0.0,
  beamSize: Int = 1,
  maxParallelThreads: Int = 1
) :
  BiRNNParser<
    ArcEagerSpineState,
    ArcEagerSpineTransition,
    GroupedDenseFeaturesErrors,
    GroupedDenseFeatures,
    MPSupportStructure,
    BiRNNArcEagerSpineParserModel>
  (
    model = model,
    wordDropoutCoefficient = wordDropoutCoefficient,
    posDropoutCoefficient = posDropoutCoefficient,
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads
  )
{

  /**
   * @return the TransitionSystem used in this parser
   */
  override fun buildTransitionSystem() = ArcEagerSpine()

  /**
   * @return the ActionsGenerator used in this parser
   */
  override fun buildActionsGenerator() = ActionsGenerator.Labeled<ArcEagerSpineState, ArcEagerSpineTransition>(
    deprels = this.model.corpusDictionary.deprelTags.getElementsReversedSet().groupBy { it.direction })

  /**
   * @return the ActionsScorer used in this parser
   */
  override fun buildActionsScorer() = SimpleArcEagerSpineActionsScorer(
    network = this.model.actionsScorerNetworkModel,
    optimizer = MultiPredictionOptimizer(
      model = this.model.actionsScorerNetworkModel,
      updateMethod = ADAMMethod(stepSize = 0.001)),
    deprelTags = this.model.corpusDictionary.deprelTags)

  /**
   * @return the SupportStructureFactory used in this parser
   */
  override fun buildSupportStructureFactory() = MPStructureFactory(model = this.model.actionsScorerNetworkModel)

  /**
   * @return the FeaturesExtractor used in this parser
   */
  override fun buildFeaturesExtractor() = ArcEagerSpineEmbeddingsFeaturesExtractor()

  /**
   * @return the BestActionSelector used in this parser during greedy decoding
   */
  override fun buildBestActionSelector() = HighestScoreActionSelector<
    ArcEagerSpineState,
    ArcEagerSpineTransition,
    DenseItem,
    TokensEmbeddingsContext>()

  /**
   * @return the MultiActionsSelector used in this parser during beam decoding
   */
  override fun buildMultiActionsSelector() = MultliActionsSelectorByScore<
    ArcEagerSpineState,
    ArcEagerSpineTransition,
    DenseItem,
    TokensEmbeddingsContext>()

  /**
   * Callback called before applying an action.
   */
  override fun beforeApplyAction(action: Transition<ArcEagerSpineTransition, ArcEagerSpineState>.Action,
                                 context: TokensEmbeddingsContext) = Unit
}
