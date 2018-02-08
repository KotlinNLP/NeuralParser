/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.helpers.Trainer
import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.syntaxdecoder.transitionsystem.oracle.OracleFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.modules.bestactionselector.BestActionSelector
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.ActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.SyntaxDecoderTrainer
import com.kotlinnlp.syntaxdecoder.context.InputContext
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State
import com.kotlinnlp.syntaxdecoder.context.items.StateItem
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure

/**
 * The training helper of the [TransitionBasedParser].
 *
 * @param neuralParser a neural parser
 * @param actionsErrorsSetter the actions errors setter
 * @param oracleFactory the oracle factory
 * @param bestActionSelector the best action selector
 * @param batchSize the size of the batches of sentences
 * @param epochs the number of training epochs
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param minRelevantErrorsCountToUpdate the min count of relevant errors needed to update the neural parser (default 1)
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
abstract class TransitionBasedTrainer<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType : InputContext<InputContextType, ItemType>,
  ItemType : StateItem<ItemType, *, *>,
  FeaturesErrorsType : FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  out SupportStructureType : DecodingSupportStructure,
  ModelType: TransitionBasedParserModel>
(
  private val neuralParser: TransitionBasedParser<StateType, TransitionType, InputContextType, ItemType, FeaturesErrorsType,
    FeaturesType, SupportStructureType, ModelType>,
  actionsErrorsSetter: ActionsErrorsSetter<StateType, TransitionType, ItemType, InputContextType>,
  oracleFactory: OracleFactory<StateType, TransitionType>,
  bestActionSelector: BestActionSelector<StateType, TransitionType, ItemType, InputContextType>,
  private val batchSize: Int,
  private val epochs: Int,
  private val validator: Validator?,
  private val modelFilename: String,
  private val minRelevantErrorsCountToUpdate: Int = 1,
  private val verbose: Boolean = true
) : Trainer(
  neuralParser = neuralParser,
  batchSize = batchSize,
  epochs = epochs,
  validator = validator,
  modelFilename = modelFilename,
  minRelevantErrorsCountToUpdate = minRelevantErrorsCountToUpdate,
  verbose = verbose
) {

  /**
   * The transition system trainer of the [neuralParser].
   */
  private val syntaxDecoderTrainer = SyntaxDecoderTrainer(
    syntaxDecoder = this.neuralParser.syntaxDecoder,
    actionsErrorsSetter = actionsErrorsSetter,
    bestActionSelector = bestActionSelector,
    oracleFactory = oracleFactory
  )

  /**
   * @return the count of the relevant errors
   */
  override fun getRelevantErrorsCount(): Int = this.syntaxDecoderTrainer.relevantErrorsCount

  /**
   * Train the Transition System with the given [sentence].
   *
   * @param sentence a sentence
   * @param goldPOSSentence an optional sentence with gold annotated POS in its dependency tree
   */
  override fun trainSentence(sentence: Sentence, goldPOSSentence: Sentence?) {

    val context: InputContextType = this.neuralParser.buildContext(sentence, trainingMode = true)
    val goldTree: DependencyTree = checkNotNull(sentence.dependencyTree) {
      "The gold dependency tree of a sentence cannot be null during the training."
    }

    this.beforeSentenceLearning(context)

    this.syntaxDecoderTrainer.learn(
      context = context,
      goldDependencyTree = goldTree,
      propagateToInput = true,
      beforeApplyAction = this::beforeApplyAction
    )

    this.afterSentenceLearning(context)
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.syntaxDecoderTrainer.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.syntaxDecoderTrainer.newEpoch()
  }

  /**
   * Update the [neuralParser].
   */
  override fun update() {

    this.syntaxDecoderTrainer.update()

    this.updateNeuralComponents()
  }

  /**
   * Update the neural components of the parser.
   */
  abstract protected fun updateNeuralComponents()

  /**
   * Callback called before the learning of a sentence.
   */
  abstract protected fun beforeSentenceLearning(context: InputContextType)

  /**
   * Callback called after the learning of a sentence.
   */
  abstract protected fun afterSentenceLearning(context: InputContextType)

  /**
   * Callback called before applying an action.
   */
  abstract protected fun beforeApplyAction(
    action: Transition<TransitionType, StateType>.Action,
    context: InputContextType)
}
