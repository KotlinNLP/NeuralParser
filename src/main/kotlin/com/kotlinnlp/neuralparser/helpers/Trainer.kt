/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.utils.Timer
import com.kotlinnlp.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
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
import java.io.File
import java.io.FileOutputStream

/**
 * The training helper of the [NeuralParser].
 *
 * @param neuralParser a neural parser
 * @param actionsErrorsSetter the actions errors setter
 * @param oracleFactory the oracle factory
 * @param epochs the number of training epochs
 * @param bestActionSelector the best action selector
 * @param batchSize the size of the batches of sentences
 * @param minRelevantErrorsCountToUpdate the min number of relevant errors needed to update the neural parser
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
abstract class Trainer<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType : InputContext<InputContextType, ItemType>,
  ItemType : StateItem<ItemType, *, *>,
  FeaturesErrorsType : FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  out SupportStructureType : DecodingSupportStructure,
  ModelType: NeuralParserModel>
(
  private val neuralParser: NeuralParser<StateType, TransitionType, InputContextType, ItemType, FeaturesErrorsType,
    FeaturesType, SupportStructureType, ModelType>,
  actionsErrorsSetter: ActionsErrorsSetter<StateType, TransitionType, ItemType, InputContextType>,
  oracleFactory: OracleFactory<StateType, TransitionType>,
  private val epochs: Int,
  bestActionSelector: BestActionSelector<StateType, TransitionType, ItemType, InputContextType>,
  private val batchSize: Int,
  private val minRelevantErrorsCountToUpdate: Int,
  private val validator: Validator?,
  private val modelFilename: String,
  private val verbose: Boolean = true
) {

  /**
   * A timer to track the elapsed time.
   */
  private var timer = Timer()

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
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = -1.0 // -1 used as init value (all accuracy values are in the range [0.0, 1.0])

  /**
   * Check requirements.
   */
  init {
    require(this.minRelevantErrorsCountToUpdate > 0) { "minRelevantErrorsCountToUpdate must be > 0" }
  }

  /**
   * Train the Transition System with the given sentences.
   *
   * @param trainingSentences the sentences used to train the ActionsScorer
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  fun train(trainingSentences: ArrayList<Sentence>,
            shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743)) {

    (0 until this.epochs).forEach { i ->

      this.logTrainingStart(epochIndex = i)

      this.newEpoch()
      this.trainEpoch(trainingSentences = trainingSentences, shuffler = shuffler)

      this.logTrainingEnd()

      if (this.validator != null) {

        this.logValidationStart()

        this.validateAndSaveModel()

        this.logValidationEnd()
      }
    }
  }

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

  /**
   * Train the Transition System.
   *
   * @param trainingSentences the training sentences
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  private fun trainEpoch(trainingSentences: ArrayList<Sentence>, shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(trainingSentences.size)

    this.newBatch()

    ExamplesIndices(trainingSentences.size, shuffler = shuffler).forEach { sentenceIndex ->

      progress.tick()

      this.trainSentence(sentence = trainingSentences[sentenceIndex])

      if (this.syntaxDecoderTrainer.relevantErrorsCount >= this.minRelevantErrorsCountToUpdate) {
        this.syntaxDecoderTrainer.update()
        this.update()
        this.newBatch()
      }
    }
  }

  /**
   * Train the Transition System with the given [sentence].
   *
   * @param sentence a sentence
   */
  private fun trainSentence(sentence: Sentence) {

    val context: InputContextType = this.neuralParser.buildContext(sentence, trainingMode = true)
    val goldTree: DependencyTree = checkNotNull(sentence.dependencyTree) {
      "The gold dependency tree of a sentence was null during its training."
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
   * Validate the [neuralParser] with the validation helper and save the best model.
   * The [validator] is required to be not null.
   */
  private fun validateAndSaveModel() {

    val stats: Statistics = this.validator!!.evaluate()

    println("\n$stats")

    if (stats.noPunctuation.uas.perc > this.bestAccuracy) {

      this.neuralParser.model.dump(FileOutputStream(File(this.modelFilename)))
      println("\nNEW BEST ACCURACY! Model saved to \"${this.modelFilename}\"")

      this.bestAccuracy = stats.noPunctuation.uas.perc
    }
  }

  /**
   * Beat the occurrence of a new batch.
   */
  private fun newBatch() {
    this.syntaxDecoderTrainer.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  private fun newEpoch() {
    this.syntaxDecoderTrainer.newEpoch()
  }

  /**
   * Update the [neuralParser]
   */
  abstract protected fun update()

  /**
   * Log when training starts.
   *
   * @param epochIndex the current epoch index
   */
  private fun logTrainingStart(epochIndex: Int) {

    if (this.verbose) {

      this.timer.reset()

      println("\nEpoch ${epochIndex + 1} of ${this.epochs}")
      println("\nStart training...")
    }
  }

  /**
   * Log when training ends.
   */
  private fun logTrainingEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))
    }
  }

  /**
   * Log when validation starts.
   */
  private fun logValidationStart() {

    if (this.verbose) {
      this.timer.reset()
      println() // new line
    }
  }

  /**
   * Log when validation ends.
   */
  private fun logValidationEnd() {

    if (this.verbose) {
      println("Elapsed time: %s".format(this.timer.formatElapsedTime()))
    }
  }
}
