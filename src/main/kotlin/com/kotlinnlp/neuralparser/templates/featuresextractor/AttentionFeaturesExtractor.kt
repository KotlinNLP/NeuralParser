/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.featuresextractor

import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEncodingContext
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.AttentionDecodingSupportStructure
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetwork
import com.kotlinnlp.simplednn.deeplearning.attentionnetwork.AttentionNetworkParameters
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractor
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractorTrainable
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext

/**
 * The [FeaturesExtractorTrainable] that extracts features encoding them in a recurrent system that uses an Attention
 * Network followed by an RNN.
 *
 * @param actionsVectors the encoding vectors of the actions
 * @param actionsVectorsOptimizer the optimizer of the [actionsVectors]
 * @param actionDecoderOptimizer the optimizer of the RNN used to decode the action features
 * @param actionAttentionNetworkOptimizer the optimizer of the attention network used to decode the action features
 * @param featuresEncodingSize the size of the features encoding
 */
abstract class AttentionFeaturesExtractor<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>>
(
  private val actionsVectors: ActionsVectorsMap,
  private val actionsVectorsOptimizer: ActionsVectorsOptimizer,
  private val actionDecoderOptimizer: ParamsOptimizer<NetworkParameters>,
  private val actionAttentionNetworkOptimizer: ParamsOptimizer<AttentionNetworkParameters>,
  featuresEncodingSize: Int
) : FeaturesExtractorTrainable<
  StateType,
  TransitionType,
  InputContextType,
  DenseItem,
  DenseFeatures,
  AttentionDecodingSupportStructure>() {

  /**
   * The actions embeddings map key of the transition of this action.
   */
  abstract protected val Transition<TransitionType, StateType>.key: Int

  /**
   * The actions embeddings map key of the deprel of this action.
   */
  abstract protected val Transition<TransitionType, StateType>.Action.deprelKey: Int

  /**
   * The actions embeddings map key of the POS tag of this action.
   */
  abstract protected val Transition<TransitionType, StateType>.Action.posTagKey: Int

  /**
   * The size of the action encoding vectors.
   * It is the concatenation of 3 action embeddings (T+P+D).
   */
  private val actionEncodingSize: Int = 3 * this.actionsVectors.size

  /**
   * The list of actions applied to the current state.
   */
  private val appliedActions = mutableListOf<Transition<TransitionType, StateType>.Action?>()

  /**
   * The list of Attention Networks used for the current action.
   */
  private val usedAttentionNetworks = mutableListOf<AttentionNetwork<DenseNDArray>>()

  /**
   * The structure used to store the params errors of the Attention Network during the backward.
   */
  private val attentionNetworkParamsErrors: AttentionNetworkParameters = this.attentionNetwork.model.copy()

  /**
   * The list of the errors of all the features extracted during the decoding of the current sentence.
   */
  private val featuresErrorsList = mutableListOf<DenseNDArray>()

  /**
   * The zeros array used as features encoding errors when no relevant errors are found.
   */
  private val featuresEncodingZerosErrors = DenseNDArrayFactory.zeros(Shape(featuresEncodingSize))

  /**
   * The RNN used to decode the action features.
   * It is saved during the extract to make the backward working on it.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var actionDecoder: RecurrentNeuralProcessor<DenseNDArray>

  /**
   * The Attention Network used to decode the action features.
   * It is saved during the extract to make the backward working on it.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var attentionNetwork: AttentionNetwork<DenseNDArray>

  /**
   * The last decoding context used.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var decodingContext
    : DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>

  /**
   * The recurrent errors of the features, obtained during the last backward step from the input errors of the
   * Attention Network.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var lastRecurrentErrors: DenseNDArray

  /**
   * Extract features using the given [decodingContext] amd [supportStructure].
   *
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the extracted features
   */
  override fun extract(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: AttentionDecodingSupportStructure
  ): DenseFeatures {

    if (decodingContext.extendedState.context.trainingMode) {
      this.checkMissingBackwardCall()
    }

    this.actionDecoder = supportStructure.actionDecoder
    this.decodingContext = decodingContext

    return DenseFeatures(array = supportStructure.actionDecoder.forward(
      featuresArray = this.getActionFeatures(decodingContext = decodingContext, supportStructure = supportStructure),
      firstState = decodingContext.extendedState.appliedActions.isEmpty()))
  }

  /**
   * Callback called after the learning of a sentence.
   */
  fun afterSentenceLearning() {

    this.checkMissingBackwardCall()

    this.backwardEncoderDecoder()

    this.resetHistory()
  }

  /**
   * Backward errors through this [FeaturesExtractor], starting from the errors of the features contained in the given
   * [decodingContext].
   * Errors are required to be already set into the given features.
   *
   * @param decodingContext the decoding context that contains extracted features with their errors
   * @param supportStructure the decoding support structure
   * @param propagateToInput a Boolean indicating whether errors must be propagated to the input items
   */
  override fun backward(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem,
      DenseFeatures>,
    supportStructure: AttentionDecodingSupportStructure,
    propagateToInput: Boolean
  ) {
    if (propagateToInput) {
      this.featuresErrorsList.add(decodingContext.features.errors.array)
    }
  }

  /**
   * Beat the occurrence of a new example.
   */
  override fun newExample() {
    this.actionDecoderOptimizer.newExample()
    this.actionAttentionNetworkOptimizer.newExample()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.actionDecoderOptimizer.newBatch()
    this.actionAttentionNetworkOptimizer.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.actionDecoderOptimizer.newEpoch()
    this.actionAttentionNetworkOptimizer.newEpoch()
  }

  /**
   * Update the trainable components of this [FeaturesExtractor].
   */
  override fun update() {
    this.actionsVectorsOptimizer.update()
    this.actionDecoderOptimizer.update()
    this.actionAttentionNetworkOptimizer.update()
  }

  /**
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the features for the Actions Scorer, extracted from the given [decodingContext]
   */
  private fun getActionFeatures(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: AttentionDecodingSupportStructure
  ): DenseNDArray {

    val appliedActions = decodingContext.extendedState.appliedActions
    val attentionNetwork = this.getAttentionNetwork(
      supportStructure = supportStructure,
      firstState = appliedActions.isEmpty(),
      trainingMode = decodingContext.extendedState.context.trainingMode)

    val lastAppliedActionEncoding: DenseNDArray = this.getLastAppliedActionEncoding(
      appliedActions = appliedActions,
      trainingMode = decodingContext.extendedState.context.trainingMode)

    val encodedStatesAttention: DenseNDArray = attentionNetwork.forward(
      inputSequence = this.buildAttentionInputSequence(context = decodingContext.extendedState.context))

    return supportStructure.actionDecoder.forward(
      featuresArray = concatVectorsV(lastAppliedActionEncoding, encodedStatesAttention),
      firstState = appliedActions.isEmpty())
  }

  /**
   * @param appliedActions the list of applied actions, taken from the decoding context
   * @param trainingMode a boolean indicating if the parser is being training
   *
   * @return the encoding vector of the last applied action
   */
  private fun getLastAppliedActionEncoding(appliedActions: List<Transition<TransitionType, StateType>.Action>,
                                           trainingMode: Boolean): DenseNDArray {

    return if (appliedActions.isEmpty()) { // first action

      if (trainingMode) this.appliedActions.add(null)

      this.actionsVectors.getNullVector()

    } else {

      val lastAction = appliedActions.last()

      if (trainingMode) this.appliedActions.add(lastAction)

      this.actionsVectors[lastAction.transition.key, lastAction.posTagKey, lastAction.deprelKey]
    }
  }

  /**
   * Get an available Attention Network to encode the last applied action.
   *
   * @param supportStructure the decoding support structure
   * @param firstState a boolean indicating if the current is the first state
   * @param trainingMode a boolean indicating if the parser is being training
   *
   * @return an available Attention Network
   */
  private fun getAttentionNetwork(supportStructure: AttentionDecodingSupportStructure,
                                  firstState: Boolean,
                                  trainingMode: Boolean): AttentionNetwork<DenseNDArray> {

    if (firstState) supportStructure.attentionNetworksPool.releaseAll()

    val attentionNetwork = supportStructure.attentionNetworksPool.getItem()
    this.usedAttentionNetworks.add(attentionNetwork)

    if (!trainingMode) supportStructure.attentionNetworksPool.releaseAll() // use always the first of the pool

    return attentionNetwork
  }

  /**
   * @param context the input context
   *
   * @return the input sequence of the decoding Attention Network
   */
  private fun buildAttentionInputSequence(context: InputContextType): ArrayList<AugmentedArray<DenseNDArray>> =
    ArrayList(context.items.map { AugmentedArray(values = context.getTokenEncoding(it.id)) })

  /**
   * Reset the history of the last sentence processed.
   */
  private fun resetHistory() {

    this.appliedActions.clear()
    this.usedAttentionNetworks.clear()
    this.featuresErrorsList.clear()
  }

  /**
   * Check if the backward wasn't called after the application of the last action.
   * It happens when the minimum number of relevant errors is not reached after the application of an action.
   */
  private fun checkMissingBackwardCall() {

    if (this.appliedActions.size > this.featuresErrorsList.size) {
      this.featuresErrorsList.add(this.featuresEncodingZerosErrors)
    }
  }

  /**
   * Encoder and decoder backward.
   */
  private fun backwardEncoderDecoder() {

    (0 until this.featuresErrorsList.size).reversed().forEach { i ->

      val isLastStep: Boolean = i == this.featuresErrorsList.lastIndex
      val errors = if (isLastStep)
        this.featuresErrorsList[i]
      else
        this.featuresErrorsList[i].sum(this.lastRecurrentErrors)

      this.backwardStep(outputErrors = errors, stepIndex = i)
    }

    this.actionDecoderOptimizer.accumulate(this.actionDecoder.getParamsErrors(copy = false))
  }

  /**
   *
   */
  private fun backwardStep(outputErrors: DenseNDArray, stepIndex: Int) {

    this.actionDecoder.backwardStep(outputErrors = outputErrors, propagateToInput = true)

    val errors: DenseNDArray = this.actionDecoder.getInputErrors(elementIndex = stepIndex, copy = false)
    val splitErrors: Array<DenseNDArray> = errors.splitV(
      this.actionEncodingSize,
      errors.length - this.actionEncodingSize
    )

    if (stepIndex == 0) {
      this.actionsVectorsOptimizer.accumulateNullEmbeddingsErrors(splitErrors[0])
    } else {
      this.accumulateActionEmbeddingErrors(errors = splitErrors[0], actionIndex = stepIndex)
    }

    this.backwardAttentionNetwork(
      attentionNetwork = this.usedAttentionNetworks[stepIndex],
      outputErrors = splitErrors[1])
  }

  /**
   * Backward of the given [attentionNetwork].
   *
   * @param attentionNetwork an Attention Network used to decode the current sentence
   * @param outputErrors the output errors of the given [attentionNetwork]
   */
  private fun backwardAttentionNetwork(attentionNetwork: AttentionNetwork<DenseNDArray>, outputErrors: DenseNDArray) {

    attentionNetwork.backward(
      outputErrors = outputErrors,
      paramsErrors = this.attentionNetworkParamsErrors,
      propagateToInput = true)

    this.actionAttentionNetworkOptimizer.accumulate(this.attentionNetworkParamsErrors)

    attentionNetwork.getInputErrors().forEachIndexed { i, errors ->

      val splitErrors: Array<DenseNDArray> = errors.splitV(
        this.decodingContext.extendedState.context.encodingSize,
        errors.length - this.decodingContext.extendedState.context.encodingSize
      )

      this.lastRecurrentErrors = if (i == 0) splitErrors[0] else this.lastRecurrentErrors.assignSum(splitErrors[0])

      this.decodingContext.extendedState.context.accumulateItemErrors(itemIndex = i, errors = splitErrors[1])
    }
  }

  /**
   * Accumulate errors of an action embedding.
   *
   * @param errors the errors of an action embedding
   * @param actionIndex the index of the related action
   */
  private fun accumulateActionEmbeddingErrors(errors: DenseNDArray, actionIndex: Int) {

    if (actionIndex == 0) {
      this.actionsVectorsOptimizer.accumulateNullEmbeddingsErrors(errors)

    } else {
      val action = this.appliedActions[actionIndex]!!
      this.actionsVectorsOptimizer.accumulate(
        tId = action.transition.key,
        pId = action.posTagKey,
        dId = action.deprelKey,
        errors = errors
      )
    }
  }
}
