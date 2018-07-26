/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.statistics.Statistics
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.utils.Timer
import com.kotlinnlp.simplednn.dataset.Shuffler
import com.kotlinnlp.simplednn.helpers.training.utils.ExamplesIndices
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

/**
 * The training helper of the [NeuralParser].
 *
 * @param neuralParser a neural parser
 * @param batchSize the number of sentences that compose a batch
 * @param epochs the number of training epochs
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param minRelevantErrorsCountToUpdate the min count of relevant errors needed to update the neural parser (default 1)
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
abstract class Trainer(
  private val neuralParser: NeuralParser<*>,
  private val batchSize: Int,
  private val epochs: Int,
  private val validator: Validator?,
  private val modelFilename: String,
  private val minRelevantErrorsCountToUpdate: Int = 1,
  private val verbose: Boolean = true
) {

  /**
   * A timer to track the elapsed time.
   */
  private var timer = Timer()

  /**
   * The best accuracy reached during the training.
   */
  private var bestAccuracy: Double = -1.0 // -1 used as init value (all accuracy values are in the range [0.0, 1.0])

  /**
   * Check requirements.
   */
  init {
    require(this.epochs > 0) { "The number of epochs must be > 0" }
    require(this.batchSize > 0) { "The size of the batch must be > 0" }
    require(this.minRelevantErrorsCountToUpdate > 0) { "minRelevantErrorsCountToUpdate must be > 0" }
  }

  /**
   * Train the [neuralParser] with the given sentences.
   *
   * @param trainingSentences the sentences used to train the parser
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  fun train(trainingSentences: List<CoNLLSentence>,
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
   * Train the Transition System.
   *
   * @param trainingSentences the training sentences
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  private fun trainEpoch(trainingSentences: List<CoNLLSentence>,
                         shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(trainingSentences.size)

    this.newBatch()

    ExamplesIndices(trainingSentences.size, shuffler = shuffler).forEachIndexed { i, sentenceIndex ->

      val endOfBatch: Boolean = (i + 1) % this.batchSize == 0 || i == trainingSentences.lastIndex

      progress.tick()

      val sentence: CoNLLSentence = trainingSentences[sentenceIndex]

      require(sentence.hasAnnotatedHeads()) { "The gold dependency tree of a sentence cannot be null during the evaluation." }

      this.trainSentence(sentence = ParsingSentence.fromCoNLL(sentence), goldTree = sentence.getDependencyTree())

      if (endOfBatch && this.getRelevantErrorsCount() >= this.minRelevantErrorsCountToUpdate) {
        this.update()
        this.newBatch()
      }
    }
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

  /**
   * Beat the occurrence of a new batch.
   */
  protected open fun newBatch() = Unit

  /**
   * Beat the occurrence of a new epoch.
   */
  protected open fun newEpoch() = Unit

  /**
   * Update the [neuralParser].
   */
  protected abstract fun update()

  /**
   * Train the Transition System with the given [sentence] and [goldTree].
   *
   * @param sentence a sentence
   * @param goldTree the gold dependency tree
   */
  protected abstract fun trainSentence(sentence: ParsingSentence, goldTree: DependencyTree)

  /**
   * @return the count of the relevant errors
   */
  protected abstract fun getRelevantErrorsCount(): Int
}
