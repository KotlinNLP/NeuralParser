/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.syntaxdecoder.BeamDecoder
import com.kotlinnlp.syntaxdecoder.GreedyDecoder
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.SyntaxDecoder
import com.kotlinnlp.syntaxdecoder.context.InputContext
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State
import com.kotlinnlp.syntaxdecoder.context.items.StateItem
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorer
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.BestActionSelector
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.MultiActionsSelector
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractor
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.SupportStructureFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.ActionsGenerator
import com.kotlinnlp.syntaxdecoder.transitionsystem.TransitionSystem

/**
 * A transition based Neural Parser.
 *
 * If the beamSize is 1 then a [GreedyDecoder] is used, a [BeamDecoder] otherwise.
 *
 * @property model the model of this neural parser
 * @param beamSize the max size of the beam (a [GreedyDecoder] is used if it is 1, a [BeamDecoder] otherwise)
 * @param maxParallelThreads the max number of threads that can run in parallel (ignored if beamSize is 1)
 */
abstract class TransitionBasedParser<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType : InputContext<InputContextType, ItemType>,
  ItemType : StateItem<ItemType, *, *>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  SupportStructureType : DecodingSupportStructure,
  out ModelType: TransitionBasedParserModel>
(
  override val model: ModelType,
  beamSize: Int,
  maxParallelThreads: Int
) :
  NeuralParser<ModelType> {

  /**
   * The syntax decoder of this parser.
   */
  val syntaxDecoder: SyntaxDecoder<StateType, TransitionType, InputContextType, ItemType, FeaturesType,
    SupportStructureType>
    = if (beamSize == 1) this.buildGreedyDecoder() else this.buildBeamDecoder(beamSize, maxParallelThreads)

  /**
   * Parse a sentence, giving its dependency tree.
   *
   * @param sentence a [Sentence]
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: Sentence): DependencyTree {

    return this.syntaxDecoder.decode(
      context = this.buildContext(sentence = sentence, trainingMode = false),
      beforeApplyAction = this::beforeApplyAction)
  }

  /**
   * @return the [TransitionSystem] used in this parser
   */
  protected abstract fun buildTransitionSystem(): TransitionSystem<StateType, TransitionType>

  /**
   * @return the [ActionsGenerator] used in this parser
   */
  protected abstract fun buildActionsGenerator(): ActionsGenerator<StateType, TransitionType>

  /**
   * @return the [ActionsScorer] used in this parser
   */
  protected abstract fun buildActionsScorer(): ActionsScorer<StateType, TransitionType, InputContextType, ItemType,
    FeaturesType, SupportStructureType>

  /**
   * @return the [SupportStructureFactory] used in this parser
   */
  protected abstract fun buildSupportStructureFactory(): SupportStructureFactory<SupportStructureType>

  /**
   * @return the [FeaturesExtractor] used in this parser
   */
  protected abstract fun buildFeaturesExtractor(): FeaturesExtractor<StateType, TransitionType, InputContextType,
    ItemType, FeaturesType, SupportStructureType>

  /**
   * @return the [BestActionSelector] used in this parser during greedy decoding
   */
  protected abstract fun buildBestActionSelector(): BestActionSelector<StateType, TransitionType, ItemType,
    InputContextType>

  /**
   * @return the [MultiActionsSelector] used in this parser during beam decoding
   */
  protected abstract fun buildMultiActionsSelector(): MultiActionsSelector<StateType, TransitionType, ItemType,
    InputContextType>

  /**
   * @param sentence a sentence
   * @param trainingMode a Boolean indicating whether the parser is being trained
   *
   * @return the [InputContext] related to the given [sentence]
   */
  abstract fun buildContext(sentence: Sentence, trainingMode: Boolean): InputContextType

  /**
   * Callback called before applying an action.
   */
  protected abstract fun beforeApplyAction(
    action: Transition<TransitionType, StateType>.Action,
    context: InputContextType)

  /**
   * @return a [GreedyDecoder] for this parser
   */
  private fun buildGreedyDecoder() = GreedyDecoder(
    transitionSystem = this.buildTransitionSystem(),
    actionsGenerator = this.buildActionsGenerator(),
    featuresExtractor = this.buildFeaturesExtractor(),
    actionsScorer = this.buildActionsScorer(),
    bestActionSelector = this.buildBestActionSelector(),
    supportStructureFactory = this.buildSupportStructureFactory(),
    scoreAccumulatorFactory = this.model.scoreAccumulatorFactory)

  /**
   * @param beamSize the max size of the beam
   * @param maxParallelThreads the max number of threads that can run in parallel
   *
   * @return a [BeamDecoder] for this parser
   */
  private fun buildBeamDecoder(beamSize: Int, maxParallelThreads: Int) = BeamDecoder(
    beamSize = beamSize,
    maxParallelThreads = maxParallelThreads,
    transitionSystem = this.buildTransitionSystem(),
    actionsGenerator = this.buildActionsGenerator(),
    featuresExtractor = this.buildFeaturesExtractor(),
    actionsScorer = this.buildActionsScorer(),
    multiActionsSelector = this.buildMultiActionsSelector(),
    supportStructureFactory = this.buildSupportStructureFactory(),
    scoreAccumulatorFactory = this.model.scoreAccumulatorFactory)
}
