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
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.feedforward.FeedforwardLayerStructure
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
 * @param transformLayerOptimizer the optimizer of the transform layers of the Attention Network
 * @param actionEncodingRNNOptimizer the optimizer of the RNN used to encode the Actions Scorer features
 * @param stateAttentionNetworkOptimizer the optimizer of the attention network used to encode the state
 * @param featuresEncodingSize the size of the features encoding
 */
abstract class AttentionFeaturesExtractor<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>>
(
  private val actionsVectors: ActionsVectorsMap,
  private val actionsVectorsOptimizer: ActionsVectorsOptimizer,
  private val transformLayerOptimizer: ParamsOptimizer<FeedforwardLayerParameters>,
  private val actionEncodingRNNOptimizer: ParamsOptimizer<NetworkParameters>,
  private val stateAttentionNetworkOptimizer: ParamsOptimizer<AttentionNetworkParameters>,
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
   * The list of the errors of all the action encodings obtained during the encoding of the current sentence.
   */
  private val actionEncodingErrorsList = mutableListOf<DenseNDArray>()

  /**
   * The zeros array used as null action encoding.
   */
  private val actionEncodingZerosArray = DenseNDArrayFactory.zeros(Shape(featuresEncodingSize))

  /**
   * The structure used to store the params errors of the transform layers during the backward.
   */
  private lateinit var transformLayerParamsErrors: FeedforwardLayerParameters

  /**
   * The structure used to store the params errors of the Attention Network during the backward.
   */
  private lateinit var attentionNetworkParamsErrors: AttentionNetworkParameters

  /**
   * The transform layers used during the last forward.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var usedTransformLayers: List<FeedforwardLayerStructure<DenseNDArray>>

  /**
   * The RNN used to encode the features.
   * It is saved during the extract to make the backward working on it.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var actionRNNEncoder: RecurrentNeuralProcessor<DenseNDArray>

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

    this.actionRNNEncoder = supportStructure.actionRNNEncoder
    this.decodingContext = decodingContext

    return DenseFeatures(array = supportStructure.actionRNNEncoder.forward(
      featuresArray = this.getActionRNNInputFeatures(decodingContext = decodingContext, supportStructure = supportStructure),
      firstState = decodingContext.extendedState.appliedActions.isEmpty()))
  }

  /**
   * Callback called after the learning of a sentence.
   */
  fun afterSentenceLearning() {

    this.checkMissingBackwardCall()

    this.backwardEncoders()

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
      this.actionEncodingErrorsList.add(decodingContext.features.errors.array)
    }
  }

  /**
   * Beat the occurrence of a new example.
   */
  override fun newExample() {
    this.actionEncodingRNNOptimizer.newExample()
    this.stateAttentionNetworkOptimizer.newExample()
    this.transformLayerOptimizer.newExample()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.actionEncodingRNNOptimizer.newBatch()
    this.stateAttentionNetworkOptimizer.newBatch()
    this.transformLayerOptimizer.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.actionEncodingRNNOptimizer.newEpoch()
    this.stateAttentionNetworkOptimizer.newEpoch()
    this.transformLayerOptimizer.newEpoch()
  }

  /**
   * Update the trainable components of this [FeaturesExtractor].
   */
  override fun update() {
    this.actionsVectorsOptimizer.update()
    this.transformLayerOptimizer.update()
    this.actionEncodingRNNOptimizer.update()
    this.stateAttentionNetworkOptimizer.update()
  }

  /**
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the input features for the action RNN encoder
   */
  private fun getActionRNNInputFeatures(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: AttentionDecodingSupportStructure
  ): DenseNDArray {

    val context: InputContextType = decodingContext.extendedState.context
    val appliedActions = decodingContext.extendedState.appliedActions
    val isFirstState: Boolean = appliedActions.isEmpty()

    val attentionNetwork = this.getAttentionNetwork(
      supportStructure = supportStructure,
      firstState = isFirstState,
      trainingMode = context.trainingMode)

    val lastAppliedActionEncoding: DenseNDArray = this.getLastAppliedActionEncoding(
      appliedActions = appliedActions,
      trainingMode = context.trainingMode)

    val encodedState: DenseNDArray = attentionNetwork.forward(
      inputSequence = ArrayList(context.items.map { AugmentedArray(values = context.getTokenEncoding(it.id)) }),
      attentionSequence = this.buildAttentionSequence(
        supportStructure = supportStructure,
        context = context,
        isFirstState = isFirstState))

    return concatVectorsV(lastAppliedActionEncoding, encodedState)
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
   * Get an available Attention Network to encode the state.
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

    if (firstState) supportStructure.stateAttentionNetworksPool.releaseAll()

    val attentionNetwork = supportStructure.stateAttentionNetworksPool.getItem()

    if (trainingMode) {
      this.usedAttentionNetworks.add(attentionNetwork)
    } else {
      supportStructure.stateAttentionNetworksPool.releaseAll() // use always the first of the pool
    }

    return attentionNetwork
  }

  /**
   * @param supportStructure the decoding support structure
   * @param context the input context
   * @param isFirstState a boolean indicating if the current is the first state of the sentence
   *
   * @return the input of the transform layer
   */
  private fun buildAttentionSequence(supportStructure: AttentionDecodingSupportStructure,
                                     context: InputContextType,
                                     isFirstState: Boolean): ArrayList<DenseNDArray> {

    this.initTransformLayers(size = context.items.size, supportStructure = supportStructure)

    val lastActionEncoding: DenseNDArray = if (isFirstState)
      this.actionEncodingZerosArray
    else
      supportStructure.actionRNNEncoder.getOutput(copy = true)

    return ArrayList(context.items.mapIndexed { i, item ->

      this.usedTransformLayers[i].setInput(concatVectorsV(context.getTokenEncoding(item.id), lastActionEncoding))
      this.usedTransformLayers[i].forward()

      this.usedTransformLayers[i].outputArray.values
    })
  }

  /**
   * Initialize the list of transform layers to use for the current decoding.
   *
   * @param size the number of layers to initialize
   */
  private fun initTransformLayers(size: Int, supportStructure: AttentionDecodingSupportStructure) {

    supportStructure.transformLayersPool.releaseAll()

    this.usedTransformLayers = List(size = size, init = { supportStructure.transformLayersPool.getItem() })
  }

  /**
   * Reset the history of the last sentence processed.
   */
  private fun resetHistory() {

    this.appliedActions.clear()
    this.usedAttentionNetworks.clear()
    this.actionEncodingErrorsList.clear()
  }

  /**
   * Check if the backward wasn't called after the application of the last action.
   * It happens when the minimum number of relevant errors is not reached after the application of an action.
   */
  private fun checkMissingBackwardCall() {

    if (this.appliedActions.size > this.actionEncodingErrorsList.size) {
      this.actionEncodingErrorsList.add(this.actionEncodingZerosArray)
    }
  }

  /**
   * Encoders backward.
   */
  private fun backwardEncoders() {

    (0 until this.actionEncodingErrorsList.size).reversed().forEach { i ->

      val isLastStep: Boolean = i == this.actionEncodingErrorsList.lastIndex
      val errors = if (isLastStep)
        this.actionEncodingErrorsList[i]
      else
        this.actionEncodingErrorsList[i].sum(this.lastRecurrentErrors)

      this.backwardStep(encodingErrors = errors, stepIndex = i)
    }

    this.actionEncodingRNNOptimizer.accumulate(this.actionRNNEncoder.getParamsErrors(copy = false))
  }

  /**
   * A single step of backward among the RNN input sequence.
   *
   * @param encodingErrors the errors of a single encoding step
   * @param stepIndex the step index
   */
  private fun backwardStep(encodingErrors: DenseNDArray, stepIndex: Int) {

    this.actionRNNEncoder.backwardStep(outputErrors = encodingErrors, propagateToInput = true)

    val errors: DenseNDArray = this.actionRNNEncoder.getInputErrors(elementIndex = stepIndex, copy = false)
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
   * @param attentionNetwork an Attention Network used to encode the current state
   * @param outputErrors the output errors of the given [attentionNetwork]
   */
  private fun backwardAttentionNetwork(attentionNetwork: AttentionNetwork<DenseNDArray>, outputErrors: DenseNDArray) {

    val transformParamsErrors: FeedforwardLayerParameters = this.getTransformParamsErrors()
    val attentionParamsErrors: AttentionNetworkParameters = this.getAttentionParamsErrors()

    attentionNetwork.backward(outputErrors = outputErrors, paramsErrors = attentionParamsErrors, propagateToInput = true)
    this.stateAttentionNetworkOptimizer.accumulate(attentionParamsErrors)

    val inputErrors = attentionNetwork.getInputErrors()

    attentionNetwork.getAttentionErrors().forEachIndexed { i, errors ->

      val transformErrors: DenseNDArray = this.backwardTransformLayer(
        layerIndex = i,
        outputErrors = errors,
        paramsErrors = transformParamsErrors)

      val splitErrors: Array<DenseNDArray> = transformErrors.splitV(
        this.decodingContext.extendedState.context.encodingSize,
        transformErrors.length - this.decodingContext.extendedState.context.encodingSize
      )

      this.decodingContext.extendedState.context.accumulateItemErrors(
        itemIndex = i,
        errors = splitErrors[0].assignSum(inputErrors[i]))

      this.lastRecurrentErrors = if (i == 0) splitErrors[1] else this.lastRecurrentErrors.assignSum(splitErrors[1])
    }
  }

  /**
   * A single transform layer backward.
   *
   * @param layerIndex the index of the layer
   * @param outputErrors the errors of the output
   * @param paramsErrors the structure in which to save the errors of the parameters
   *
   * @return the errors of the input
   */
  private fun backwardTransformLayer(layerIndex: Int,
                                     outputErrors: DenseNDArray,
                                     paramsErrors: FeedforwardLayerParameters): DenseNDArray {

    this.usedTransformLayers[layerIndex].setErrors(outputErrors)
    this.usedTransformLayers[layerIndex].backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

    this.transformLayerOptimizer.accumulate(paramsErrors)

    return this.usedTransformLayers[layerIndex].inputArray.errors
  }

  /**
   * @return the transform layers params errors
   */
  private fun getTransformParamsErrors(): FeedforwardLayerParameters = try {
    this.transformLayerParamsErrors
  } catch (e: UninitializedPropertyAccessException) {
    this.transformLayerParamsErrors = this.usedTransformLayers.last().params.copy() as FeedforwardLayerParameters
    this.transformLayerParamsErrors
  }

  /**
   * @return the Attention Network params errors
   */
  private fun getAttentionParamsErrors(): AttentionNetworkParameters = try {
    this.attentionNetworkParamsErrors
  } catch (e: UninitializedPropertyAccessException) {
    this.attentionNetworkParamsErrors = this.usedAttentionNetworks.last().model.copy()
    this.attentionNetworkParamsErrors
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
