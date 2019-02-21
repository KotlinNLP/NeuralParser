/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.lssencoder.tokensencoder.LSSTokensEncoder
import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.morphodisambiguator.language.MorphoToken
import com.kotlinnlp.morphodisambiguator.language.PropertyNames
import com.kotlinnlp.morphodisambiguator.utils.Statistics
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.core.functionalities.losses.SoftmaxCrossEntropyCalculator
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.optimizer.Optimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import com.kotlinnlp.simplednn.deeplearning.sequenceencoder.SequenceParallelEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling
import com.kotlinnlp.utils.ExamplesIndices
import com.kotlinnlp.utils.Shuffler
import com.kotlinnlp.utils.Timer
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File
import java.io.FileOutputStream

class MorphoDisambiguatorTrainer(
    private val disambiguator: MorphoDisambiguator,
    private val batchSize: Int,
    private val epochs: Int,
    private val validator: MorphoDisambiguatorValidator?,
    private val modelFilename: String,
    private val updateMethod: UpdateMethod<*> = ADAMMethod(stepSize = 0.001, beta1 = 0.9, beta2 = 0.999),
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
  }

  /**
   * Train the [MorphoDisambiguator] with the given sentences.
   *
   * @param trainingSentences the sentences used to train the disambiguator
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  fun train(trainingSentences: List<ParsingSentence>,
            goldSentences: List<PreprocessedDataset.MorphoSentence>,
            shuffler: Shuffler? = Shuffler(enablePseudoRandom = true, seed = 743)) {

    (0 until this.epochs).forEach { i ->

      this.logTrainingStart(epochIndex = i)

      this.newEpoch()
      this.trainEpoch(trainingSentences = trainingSentences, goldSentences = goldSentences, shuffler = shuffler)

      this.logTrainingEnd()

      this.validator?.apply {
        logValidationStart()
        validateAndSaveModel()
        logValidationEnd()
      }
    }
  }

  /**
   * Train the disambiguator for an epoch.
   *
   * @param trainingSentences the training sentences
   * @param shuffler a shuffle to shuffle the sentences at each epoch (can be null)
   */
  private fun trainEpoch(trainingSentences: List<ParsingSentence>,
                         goldSentences: List<PreprocessedDataset.MorphoSentence>,
                         shuffler: Shuffler?) {

    val progress = ProgressIndicatorBar(trainingSentences.size)

    this.newBatch()

    val inputIndices = ExamplesIndices(trainingSentences.size, shuffler = shuffler)
    ExamplesIndices(goldSentences.size, shuffler = shuffler)


    inputIndices.forEachIndexed { i, sentenceIndex ->

      val endOfBatch: Boolean = (i + 1) % this.batchSize == 0 || i == trainingSentences.lastIndex

      progress.tick()

      val inputSentence: ParsingSentence = trainingSentences[sentenceIndex]
      val goldSentence: PreprocessedDataset.MorphoSentence = goldSentences[sentenceIndex]

      this.trainSentence(
          sentence = inputSentence,
          goldSentence = goldSentence)

      if (endOfBatch ) {
        this.update()
        this.newBatch()
      }
    }

  }

  /**
   * Validate the [MorphoDisambiguator] with the validation helper and save the best model.
   * The [validator] is required to be not null.
   */
  private fun validateAndSaveModel() {

    val stats: Statistics = this.validator!!.evaluate()

    println("\n$stats")

    stats.reset()

    if (true) { // todo condition

      this.disambiguator.model.dump(FileOutputStream(File(this.modelFilename)))
      println("\nNEW BEST ACCURACY! Model saved to \"${this.modelFilename}\"")

    }
  }

  /**
   * The epoch counter.
   */
  private var epochCount: Int = 0


  /**
   * Beat the occurrence of a new batch.
   */
  fun newBatch() {
    if (this.updateMethod is BatchScheduling) this.updateMethod.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  fun newEpoch() {

    if (this.updateMethod is EpochScheduling) this.updateMethod.newEpoch()

    this.epochCount++
  }

  /**
   * Update the model parameters.
   */
  fun update() {
    this.optimizers.forEach { it?.update() }
  }

  /**
   * The encoder of the Latent Syntactic Structure.
   */
  private val tokensEncoder = LSSTokensEncoder(this.disambiguator.model.lssModel, useDropout = false)

  /**
   * The BIRNN encoder on top of the Lss Tokens Encoder
   */

  private val biRNNEncoder = BiRNNEncoder<DenseNDArray>(
      network = this.disambiguator.model.biRNNEncoderModel,
      propagateToInput = true,
      useDropout = false)

  /**
   * The sequence parallel encoder on top of biRNN encoder
   */
  private val outputEncoder = SequenceParallelEncoder<DenseNDArray> (
      model = this.disambiguator.model.sequenceParallelEncoderModel,
      propagateToInput = true,
      useDropout = false)

  /**
   * The optimizer of the heads encoder.
   */
  private val biRNNEncoderOptimizer: ParamsOptimizer<BiRNNParameters> = ParamsOptimizer(
      params = this.disambiguator.model.biRNNEncoderModel.model, updateMethod = this.updateMethod)


  /**
   * The optimizer of the sequence parallel encoder.
   */
  private val outputEncoderOptimizer = ParamsOptimizer (
      params = this.disambiguator.model.sequenceParallelEncoderModel.params, updateMethod = this.updateMethod)


  /**
   * Group the optimizers all together.
   */
  private val optimizers: List<Optimizer<*>?> = listOf(
      this.biRNNEncoderOptimizer,
      this.outputEncoderOptimizer)


  /**
   * Method to call before learning a new sentence.
   */
  private fun beforeSentenceLearning() {
    if (this.updateMethod is ExampleScheduling) this.updateMethod.newExample()
  }

  /**
   * Train the disambiguator with the given [sentence] and gold Morphologies.
   *
   * @param sentence the sentence
   */
  private fun trainSentence(sentence: ParsingSentence, goldSentence: PreprocessedDataset.MorphoSentence) {

    this.beforeSentenceLearning()

    val lss: List<DenseNDArray> = this.tokensEncoder.forward(sentence)

    val biRNNEncodedSentence: List<DenseNDArray> = this.biRNNEncoder.forward(lss)

    val parallelOutput: List<List<DenseNDArray>> = this.outputEncoder.forward(biRNNEncodedSentence)

    val outputErrors: List<List<DenseNDArray>> = this.calculateLoss(
        predictions = parallelOutput,
        goldMorphologies = goldSentence
    )

    this.propagateErrors(outputErrors)

  }

  /**
   * Propagate the errors through the encoders.
   *
   * @param outputErrors the latent heads errors
   */
  private fun propagateErrors(outputErrors: List<List<DenseNDArray>>) {

    this.outputEncoder.backward(outputErrors = outputErrors)

    this.biRNNEncoder.backward(outputErrors = this.outputEncoder.getInputErrors(copy = false))

    this.outputEncoderOptimizer.accumulate(paramsErrors = this.outputEncoder.getParamsErrors(copy = false))

    this.biRNNEncoderOptimizer.accumulate(paramsErrors = this.biRNNEncoder.getParamsErrors(copy = false))

  }

  /**
   * Calculate the errors for a prediction given gold index, using SOftmax Cross Entropy loss
   * @param prediction an array given by a neural network
   * @param goldIndex the index of the correct class
   */
  private fun getPredictionErrors(prediction: DenseNDArray, goldIndex: Int): DenseNDArray =
      SoftmaxCrossEntropyCalculator().calculateErrors(output = prediction, goldIndex = goldIndex)

  /**
   * Return the errors of a given labeler predictions, respect to a gold dependency tree.
   * Errors are calculated comparing the last predictions done with the given gold grammatical configurations.
   *
   * @param predictions the current network predictions
   * @param goldMorphologies the gold morphologies of the sentence
   *
   * @return a list of predictions errors
   */
  private fun calculateLoss(predictions: List<List<DenseNDArray>>,
                            goldMorphologies: PreprocessedDataset.MorphoSentence): List<List<DenseNDArray>> {

    val sentenceErrorsList = mutableListOf<List<DenseNDArray>>()

    predictions.forEachIndexed { tokenIndex, tokenPredictions ->

      val tokenGoldMorphologies: MorphoToken = goldMorphologies.tokens[tokenIndex]

      val outputErrors = mutableListOf<DenseNDArray>()

      tokenPredictions.forEachIndexed { propertyIndex, propertyPredicion ->
        val outputError: DenseNDArray
        val goldMorphology: String? = tokenGoldMorphologies.morphoProperties[propertyIndex]
        val goldMorphologyIndex: Int
        when (goldMorphology) {
          null -> {
            goldMorphologyIndex = 0
            outputError = this.getPredictionErrors(
                prediction = propertyPredicion,
                goldIndex = goldMorphologyIndex
            )
          }
          PropertyNames().unknownAnnotation -> {
            outputError = propertyPredicion.zerosLike()
          }
          else -> {
            goldMorphologyIndex = this.disambiguator.model.corpusMorphologies.morphologyDictionaryMap[
                PropertyNames().propertyNames[propertyIndex]]?.getId(goldMorphology)!!
            outputError = this.getPredictionErrors(
                prediction = propertyPredicion,
                goldIndex = goldMorphologyIndex
            )
          }
        }
        outputErrors.add(outputError)
      }

      sentenceErrorsList.add(outputErrors)
    }

    return sentenceErrorsList
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
}