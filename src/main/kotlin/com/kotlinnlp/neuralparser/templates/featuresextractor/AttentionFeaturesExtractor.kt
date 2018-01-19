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
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.AttentionSupportStructure
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
 * @param stateAttentionNetworkOptimizer the optimizer of the attention network used to encode the state
 * @param memoryRNNOptimizer the optimizer of the memory RNN
 * @param featuresLayerOptimizer the optimizer of the features layer
 * @param memoryEncodingSize the size of the memory encoding
 * @param featuresSize the size of the features
 */
abstract class AttentionFeaturesExtractor<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>,
  in SupportStructureType: AttentionSupportStructure>
(
  private val actionsVectors: ActionsVectorsMap,
  private val actionsVectorsOptimizer: ActionsVectorsOptimizer,
  private val transformLayerOptimizer: ParamsOptimizer<FeedforwardLayerParameters>,
  private val stateAttentionNetworkOptimizer: ParamsOptimizer<AttentionNetworkParameters>,
  private val memoryRNNOptimizer: ParamsOptimizer<NetworkParameters>,
  private val featuresLayerOptimizer: ParamsOptimizer<FeedforwardLayerParameters>,
  private val memoryEncodingSize: Int,
  featuresSize: Int
) : FeaturesExtractorTrainable<
  StateType,
  TransitionType,
  InputContextType,
  DenseItem,
  DenseFeatures,
  SupportStructureType>() {

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
   * The list of the errors of all the features extracted during the encoding of the current sentence.
   */
  private val featuresErrorsList = mutableListOf<DenseNDArray>()

  /**
   * The zeros array used as null memory encoding.
   */
  private val featuresMemoryZerosArray = DenseNDArrayFactory.zeros(Shape(this.memoryEncodingSize))

  /**
   * The zeros array used as null features.
   */
  private val featuresZerosArray = DenseNDArrayFactory.zeros(Shape(featuresSize))

  /**
   * The structure used to store the params errors of the transform layers during the backward.
   */
  private lateinit var transformLayerParamsErrors: FeedforwardLayerParameters

  /**
   * The structure used to store the params errors of the Attention Network during the backward.
   */
  private lateinit var attentionNetworkParamsErrors: AttentionNetworkParameters

  /**
   * The structure used to store the params errors of the features layer during the backward.
   */
  private lateinit var featuresLayerParamsErrors: FeedforwardLayerParameters

  /**
   * The transform layers used during the last forward.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var usedTransformLayers: List<FeedforwardLayerStructure<DenseNDArray>>

  /**
   * The action memory RNN.
   * It is saved during the extract to make the backward working on it.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var actionMemoryEncoder: RecurrentNeuralProcessor<DenseNDArray>

  /**
   * The features layer.
   * It is saved during the extract to make the backward working on it.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var featuresLayer: FeedforwardLayerStructure<DenseNDArray>

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
  private lateinit var recurrentMemoryErrors: DenseNDArray

  /**
   * The recurrent errors of the state encoding, obtained during the last backward step from the input errors of the
   * RNN memory encoder.
   * (Its usage makes the training no thread safe).
   */
  private lateinit var recurrentStateEncodingErrors: DenseNDArray

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
    supportStructure: SupportStructureType
  ): DenseFeatures {

    if (decodingContext.extendedState.context.trainingMode) {
      this.checkMissingBackwardCall()
    }

    this.actionMemoryEncoder = supportStructure.actionMemoryEncoder
    this.featuresLayer = supportStructure.featuresLayer
    this.decodingContext = decodingContext

    val memoryEncoding: DenseNDArray = this.getMemoryEncoding(
      decodingContext = decodingContext,
      supportStructure = supportStructure)

    supportStructure.featuresLayer.setInput(this.getFeaturesLayerInput(
      decodingContext = decodingContext,
      supportStructure = supportStructure,
      memoryEncoding = memoryEncoding))

    supportStructure.featuresLayer.forward()

    return DenseFeatures(array = supportStructure.featuresLayer.outputArray.values)
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
    supportStructure: SupportStructureType,
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
    this.transformLayerOptimizer.newExample()
    this.stateAttentionNetworkOptimizer.newExample()
    this.memoryRNNOptimizer.newExample()
    this.featuresLayerOptimizer.newExample()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.transformLayerOptimizer.newBatch()
    this.stateAttentionNetworkOptimizer.newBatch()
    this.memoryRNNOptimizer.newBatch()
    this.featuresLayerOptimizer.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.transformLayerOptimizer.newEpoch()
    this.stateAttentionNetworkOptimizer.newEpoch()
    this.memoryRNNOptimizer.newEpoch()
    this.featuresLayerOptimizer.newEpoch()
  }

  /**
   * Update the trainable components of this [FeaturesExtractor].
   */
  override fun update() {
    this.actionsVectorsOptimizer.update()
    this.transformLayerOptimizer.update()
    this.stateAttentionNetworkOptimizer.update()
    this.memoryRNNOptimizer.update()
    this.featuresLayerOptimizer.update()
  }

  /**
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   * @param memoryEncoding the memory encoding array
   *
   * @return the input of the features layer
   */
  private fun getFeaturesLayerInput(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: AttentionSupportStructure,
    memoryEncoding: DenseNDArray
  ): DenseNDArray {

    val encodedState: DenseNDArray = this.getEncodedState(
      decodingContext = decodingContext,
      supportStructure = supportStructure,
      memoryEncoding = memoryEncoding)

    supportStructure.lastEncodedState = encodedState

    return concatVectorsV(encodedState, memoryEncoding)
  }

  /**
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   * @param memoryEncoding the memory encoding array
   *
   * @return the encoded state as output of the attention network
   */
  private fun getEncodedState(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: AttentionSupportStructure,
    memoryEncoding: DenseNDArray
  ): DenseNDArray {

    val context: InputContextType = decodingContext.extendedState.context
    val appliedActions = decodingContext.extendedState.appliedActions
    val isFirstState: Boolean = appliedActions.isEmpty()

    val attentionNetwork = this.getAttentionNetwork(
      supportStructure = supportStructure,
      firstState = isFirstState,
      trainingMode = context.trainingMode)

    return attentionNetwork.forward(
      inputSequence = ArrayList(context.items.map { AugmentedArray(values = context.getTokenEncoding(it.id)) }),
      attentionSequence = this.buildAttentionSequence(
        supportStructure = supportStructure,
        context = context,
        memoryEncoding = memoryEncoding))
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
  private fun getAttentionNetwork(supportStructure: AttentionSupportStructure,
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
   * @param memoryEncoding the memory encoding array
   *
   * @return the input of the transform layer
   */
  private fun buildAttentionSequence(supportStructure: AttentionSupportStructure,
                                     context: InputContextType,
                                     memoryEncoding: DenseNDArray): ArrayList<DenseNDArray> {

    this.initTransformLayers(size = context.items.size, supportStructure = supportStructure)

    return ArrayList(context.items.zip(this.usedTransformLayers).map { (item, layer) ->

      layer.setInput(concatVectorsV(memoryEncoding, context.getTokenEncoding(item.id)))
      layer.forward()

      layer.outputArray.values
    })
  }

  /**
   * Initialize the list of transform layers to use for the current decoding.
   *
   * @param size the number of layers to initialize
   */
  private fun initTransformLayers(size: Int, supportStructure: AttentionSupportStructure) {

    supportStructure.transformLayersPool.releaseAll()

    this.usedTransformLayers = List(size = size, init = { supportStructure.transformLayersPool.getItem() })
  }

  /**
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the actions memory encoding as output of the memory RNN
   */
  private fun getMemoryEncoding(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: AttentionSupportStructure
  ): DenseNDArray {

    val appliedActions = decodingContext.extendedState.appliedActions
    val isFirstState: Boolean = appliedActions.isEmpty()

    return if (isFirstState) {

      this.featuresMemoryZerosArray

    } else {

      val lastAppliedActionEncoding: DenseNDArray = this.getLastAppliedActionEncoding(
        appliedActions = appliedActions,
        trainingMode = decodingContext.extendedState.context.trainingMode)

      supportStructure.actionMemoryEncoder.forward(
        featuresArray = concatVectorsV(supportStructure.lastEncodedState, lastAppliedActionEncoding),
        firstState = isFirstState)
    }
  }

  /**
   * @param appliedActions the list of applied actions, taken from the decoding context
   * @param trainingMode a boolean indicating if the parser is being training
   *
   * @return the encoding vector of the last applied action
   */
  private fun getLastAppliedActionEncoding(appliedActions: List<Transition<TransitionType, StateType>.Action>,
                                           trainingMode: Boolean): DenseNDArray {

    val lastAction = appliedActions.last()

    if (trainingMode) this.appliedActions.add(lastAction)

    return this.actionsVectors[lastAction.transition.key, lastAction.posTagKey, lastAction.deprelKey]
  }

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
      this.featuresErrorsList.add(this.featuresZerosArray)
    }
  }

  /**
   * Encoders backward.
   */
  private fun backwardEncoders() {

    (0 until this.featuresErrorsList.size).reversed().forEach { stepIndex ->

      val (stateEncodingOutputPart, memoryEncodingOutputPart) = this.splitFeaturesLayerInputErrors(
        errors = this.getFeaturesLayerInputErrors(featuresErrors = this.featuresErrorsList[stepIndex])
      )

      val stateEncodingErrors: DenseNDArray = if (stepIndex == this.featuresErrorsList.lastIndex)
        stateEncodingOutputPart
      else
        stateEncodingOutputPart.assignSum(this.recurrentStateEncodingErrors)

      this.backwardAttentionNetwork(
        attentionNetwork = this.usedAttentionNetworks[stepIndex],
        outputErrors = stateEncodingErrors)

      if (stepIndex > 0) {
        this.memoryEncoderBackwardStep(
          memoryEncodingErrors = memoryEncodingOutputPart.assignSum(this.recurrentMemoryErrors),
          actionIndex = stepIndex - 1)
      }
    }

    this.memoryRNNOptimizer.accumulate(this.actionMemoryEncoder.getParamsErrors(copy = false))
  }

  /**
   * @param featuresErrors the features errors
   *
   * @return the input errors of the features layer
   */
  private fun getFeaturesLayerInputErrors(featuresErrors: DenseNDArray): DenseNDArray {

    val paramsErrors: FeedforwardLayerParameters = this.getFeaturesLayerParamsErrors()

    this.featuresLayer.setErrors(featuresErrors)
    this.featuresLayer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

    this.featuresLayerOptimizer.accumulate(paramsErrors)

    return this.featuresLayer.inputArray.errors
  }

  /**
   * @param errors the input errors of the features layer
   *
   * @return a Pair containing partitions of the state encoding and the memory encoding errors
   */
  private fun splitFeaturesLayerInputErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      errors.length - this.memoryEncodingSize,
      this.memoryEncodingSize
    )

    return Pair(splitErrors[0], splitErrors[1])
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

    val inputErrors: Array<DenseNDArray> = attentionNetwork.getInputErrors()

    attentionNetwork.getAttentionErrors().forEachIndexed { itemIndex, errors ->

      val attentionArrayErrors: DenseNDArray = this.backwardTransformLayer(
        layer = this.usedTransformLayers[itemIndex],
        outputErrors = errors,
        paramsErrors = transformParamsErrors)

      val (memoryEncodingPart, tokenEncodingPart) = this.splitAttentionErrors(attentionArrayErrors)

      this.accumulateInputErrors(
        memoryEncodingErrors = memoryEncodingPart,
        itemErrors = tokenEncodingPart.assignSum(inputErrors[itemIndex]),
        itemIndex = itemIndex)
    }
  }

  /**
   * A single transform layer backward.
   *
   * @param layer a transform layer
   * @param outputErrors the errors of the output
   * @param paramsErrors the structure in which to save the errors of the parameters
   *
   * @return the errors of the input
   */
  private fun backwardTransformLayer(layer: FeedforwardLayerStructure<DenseNDArray>,
                                     outputErrors: DenseNDArray,
                                     paramsErrors: FeedforwardLayerParameters): DenseNDArray {

    layer.setErrors(outputErrors)
    layer.backward(paramsErrors = paramsErrors, propagateToInput = true, mePropK = null)

    this.transformLayerOptimizer.accumulate(paramsErrors)

    return layer.inputArray.errors
  }

  /**
   * @param errors the errors of an encoded attention array of the Attention Network
   *
   * @return a Pair containing partitions of the memory encoding and the token encoding errors
   */
  private fun splitAttentionErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val tokenEncodingSize: Int = this.decodingContext.extendedState.context.encodingSize
    val memoryEncodingSize: Int = this.usedTransformLayers.last().inputArray.size - tokenEncodingSize

    val splitErrors: Array<DenseNDArray> = errors.splitV(memoryEncodingSize, tokenEncodingSize)

    return Pair(splitErrors[0], splitErrors[1])
  }

  /**
   * Accumulate the errors of a single input, already split in the errors of the last action encoding and the errors
   * of a token encoding.
   *
   * @param memoryEncodingErrors the errors of the memory encoding, associated to a token
   * @param itemErrors the errors of the related token
   * @param itemIndex the index of the item
   */
  private fun accumulateInputErrors(memoryEncodingErrors: DenseNDArray, itemErrors: DenseNDArray, itemIndex: Int) {

      this.decodingContext.extendedState.context.accumulateItemErrors(itemIndex = itemIndex, errors = itemErrors)

      this.recurrentMemoryErrors = if (itemIndex == 0)
        memoryEncodingErrors
      else
        this.recurrentMemoryErrors.assignSum(memoryEncodingErrors)
  }

  /**
   * A single backward step of the RNN action memory encoder.
   *
   * @param memoryEncodingErrors the errors of a single memory encoding step
   * @param actionIndex the action index
   */
  private fun memoryEncoderBackwardStep(memoryEncodingErrors: DenseNDArray, actionIndex: Int) {

    this.actionMemoryEncoder.backwardStep(outputErrors = memoryEncodingErrors, propagateToInput = true)

    val errors: DenseNDArray = this.actionMemoryEncoder.getInputErrors(elementIndex = actionIndex, copy = false)
    val (prevStateEncodingErrors, prevActionEncodingErrors) = this.splitMemoryInputErrors(errors)

    this.recurrentStateEncodingErrors = prevStateEncodingErrors

    this.accumulateActionEmbeddingErrors(errors = prevActionEncodingErrors, actionIndex = actionIndex)
  }

  /**
   * @param errors the input errors of a memory encoding
   *
   * @return a Pair containing partitions of the prev state encoding and the prev action encoding errors
   */
  private fun splitMemoryInputErrors(errors: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val splitErrors: Array<DenseNDArray> = errors.splitV(
      errors.length - this.actionEncodingSize,
      this.actionEncodingSize
    )

    return Pair(splitErrors[0], splitErrors[1])
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
   * @return the features layer params errors
   */
  private fun getFeaturesLayerParamsErrors(): FeedforwardLayerParameters = try {
    this.featuresLayerParamsErrors
  } catch (e: UninitializedPropertyAccessException) {
    this.featuresLayerParamsErrors = this.featuresLayer.params.copy() as FeedforwardLayerParameters
    this.featuresLayerParamsErrors
  }
}
