/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.featuresextractor

import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEncodingContext
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.neuralparser.utils.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.attentiverecurrentnetwork.AttentiveRecurrentNetwork
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractorTrainable
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext

/**
 * The FeaturesExtractor that extracts features concatenating the encodings of a tokens window.
 *
 * @param attentiveRecurrentNetworkUpdateMethod the update method to optimize the [attentiveRecurrentNetwork] params
 */
abstract class AttentionFeaturesExtractor<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>,
  in SupportStructureType : DecodingSupportStructure>
(
  featuresSize: Int,
  private val actionsVectorsMap: ActionsVectorsMap,
  private val actionsVectorsOptimizer: ActionsVectorsOptimizer,
  private val attentiveRecurrentNetwork: AttentiveRecurrentNetwork,
  private val attentiveRecurrentNetworkUpdateMethod: UpdateMethod<*>
) : FeaturesExtractorTrainable<
    StateType,
    TransitionType,
    InputContextType,
    DenseItem,
    DenseFeatures,
    SupportStructureType>() {

  /**
   * The zeros array used as null features.
   */
  private val featuresZerosArray = DenseNDArrayFactory.zeros(Shape(featuresSize))

  /**
   * The list of the errors of all the features extracted during the encoding of the current sentence.
   */
  private val featuresErrorsList = mutableListOf<DenseNDArray>()

  /**
   * The optimizer of the [attentiveRecurrentNetwork].
   */
  private val recurrentAttentiveNetworkOptimizer = ParamsOptimizer(
    params = this.attentiveRecurrentNetwork.model.params,
    updateMethod = this.attentiveRecurrentNetworkUpdateMethod)

  /**
   * The actions embeddings map key of the transition of this action.
   */
  abstract protected val Transition<TransitionType, StateType>.key: Int

  /**
   * The actions embeddings map key of the deprel of this action (can be null).
   */
  abstract protected val Transition<TransitionType, StateType>.Action.deprelKey: Int?

  /**
   * The actions embeddings map key of the POS tag of this action (can be null).
   */
  abstract protected val Transition<TransitionType, StateType>.Action.posTagKey: Int?

  /**
   * The current decoding context.
   */
  private lateinit var curDecodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem,
    DenseFeatures>

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
    supportStructure: SupportStructureType): DenseFeatures {

    this.curDecodingContext = decodingContext // important here
    this.checkMissingBackwardCall()

    val context = decodingContext.extendedState.context
    val appliedActions = decodingContext.extendedState.appliedActions
    val firstState = appliedActions.isEmpty()

    if (firstState) {
      this.featuresErrorsList.clear()
      this.attentiveRecurrentNetwork.setInputSequence(context.items.map { context.getTokenEncoding(it.id) })
      this.attentiveRecurrentNetwork.setInitHidden(array = concatVectorsV(
        context.getTokenEncoding(context.items.last().id).getRange(0, context.encodingSize / 2),
        context.getTokenEncoding(context.items.first().id).getRange(context.encodingSize / 2, context.encodingSize)
      ))
    }

    return DenseFeatures(this.attentiveRecurrentNetwork.forward(
      lastPredictionLabel = this.getActionEncoding(if (firstState) null else appliedActions.last()),
      trainingMode = context.trainingMode))
  }

  /**
   * Backward errors through this features extractor, starting from the errors of the features contained in the given
   * [decodingContext].
   * Errors are required to be already set into the given features.
   *
   * @param decodingContext the decoding context that contains extracted features with their errors
   * @param supportStructure the decoding support structure
   * @param propagateToInput a Boolean indicating whether errors must be propagated to the input items
   */
  override fun backward(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: SupportStructureType, propagateToInput: Boolean) {

    this.featuresErrorsList.add(decodingContext.features.errors.array.copy())
  }

  /**
   * Callback called after the learning of a sentence.
   */
  fun afterSentenceLearning() {

    if (this.featuresErrorsList.isNotEmpty()) {

      this.attentiveRecurrentNetwork.backward(this.featuresErrorsList)

      this.recurrentAttentiveNetworkOptimizer.accumulate(
        paramsErrors = this.attentiveRecurrentNetwork.getParamsErrors(copy = false),
        copy = false)

      this.recurrentAttentiveNetworkOptimizer.update()

      this.propagateErrorsToActionEncodings()
      this.propagateErrorsToItems()
    }
  }

  /**
   * Update the trainable components of this object.
   */
  override fun update() {
    this.actionsVectorsOptimizer.update()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.actionsVectorsOptimizer.newBatch()
    this.recurrentAttentiveNetworkOptimizer.newBatch()
  }

  /**
   * Beat the occurrence of a new example.
   */
  override fun newExample() {
    this.actionsVectorsOptimizer.newExample()
    this.recurrentAttentiveNetworkOptimizer.newExample()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.actionsVectorsOptimizer.newEpoch()
    this.recurrentAttentiveNetworkOptimizer.newEpoch()
  }

  /**
   * Check if the backward wasn't called after the application of the last action.
   * It happens when the minimum number of relevant errors is not reached after the application of an action.
   */
  private fun checkMissingBackwardCall() {
    if (this.curDecodingContext.extendedState.appliedActions.size > this.featuresErrorsList.size) {
      this.featuresErrorsList.add(this.featuresZerosArray)
    }
  }

  /**
   * @param action an action (can be null)
   *
   * @return the encoding vector of the given [action]
   */
  private fun getActionEncoding(action: Transition<TransitionType, StateType>.Action?): DenseNDArray =
    if (action == null)
      this.actionsVectorsMap.getNullVector()
    else
      this.actionsVectorsMap[action.transition.key, action.posTagKey, action.deprelKey]

  /**
   * Propagate the errors to the action encodings.
   */
  private fun propagateErrorsToActionEncodings() {

    val prevActionEncodingsErrors: List<DenseNDArray?> = this.attentiveRecurrentNetwork.getContextLabelsErrors()

    val appliedActions = this.curDecodingContext.extendedState.appliedActions
    val prevAppliedActions = listOf(null) + appliedActions.subList(0, appliedActions.lastIndex)

    require(prevActionEncodingsErrors.size == prevAppliedActions.size) {
      "Errors not aligned with the applied actions. Expected %d, found %d."
        .format(prevAppliedActions.size, prevActionEncodingsErrors.size)
    }

    prevAppliedActions.zip(prevActionEncodingsErrors).forEach { (action, errors) -> // the last action is ignored
      if (action != null && errors != null) { // false in the first state
        this.accumulateActionEncodingErrors(action = action, errors = errors)
      }
    }
  }

  /**
   * Accumulate errors of an action embedding.
   *
   * @param action the action
   * @param errors the errors to optimize the embeddings of the given [action]
   */
  private fun accumulateActionEncodingErrors(action: Transition<TransitionType, StateType>.Action?,
                                             errors: DenseNDArray) {

    if (action == null)
      this.actionsVectorsOptimizer.accumulateNullEmbeddingsErrors(errors)
    else
      this.actionsVectorsOptimizer.accumulate(
        tId = action.transition.key,
        pId = action.posTagKey,
        dId = action.deprelKey,
        errors = errors
      )
  }

  /**
   * Propagate the errors to the items.
   */
  private fun propagateErrorsToItems() {

    val itemsErrors: List<DenseNDArray> = this.attentiveRecurrentNetwork.getInputSequenceErrors()

    itemsErrors.forEachIndexed { itemIndex, errors ->
      this.curDecodingContext.extendedState.context.accumulateItemErrors(itemIndex = itemIndex, errors = errors)
    }

    this.propagateInitHiddenErrors()
  }

  /**
   * Propagate the errors of the init hidden array to the items.
   */
  private fun propagateInitHiddenErrors() {

    val context: InputContextType = this.curDecodingContext.extendedState.context

    val initHiddenErrors: DenseNDArray = this.attentiveRecurrentNetwork.getInitHiddenErrors()
    val halfErrorsSize: Int = initHiddenErrors.length / 2
    val halfZerosErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(halfErrorsSize))

    this.curDecodingContext.extendedState.context.accumulateItemErrors(
      itemIndex = context.items.last().id,
      errors = concatVectorsV(initHiddenErrors.getRange(0, halfErrorsSize), halfZerosErrors))

    this.curDecodingContext.extendedState.context.accumulateItemErrors(
      itemIndex = context.items.first().id,
      errors = concatVectorsV(halfZerosErrors, initHiddenErrors.getRange(halfErrorsSize, initHiddenErrors.length)))
  }
}
