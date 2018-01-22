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
import com.kotlinnlp.simplednn.deeplearning.recurrentattentivedecoder.RecurrentAttentiveNetwork
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
  private val recurrentAttentiveNetwork: RecurrentAttentiveNetwork
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
  private var curDecodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>? = null

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
    val firstState = decodingContext.extendedState.appliedActions.isEmpty()

    if (firstState) this.featuresErrorsList.clear()

    val lastActionEncoding: DenseNDArray? = if (firstState) {
      null
    } else {
      this.getActionEncoding(decodingContext.extendedState.appliedActions.last())
    }

    return DenseFeatures(this.recurrentAttentiveNetwork.forward(
      inputSequence = context.items.map { context.getTokenEncoding(it.id) },
      lastPredictionLabel = lastActionEncoding,
      firstState = firstState))
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

      this.recurrentAttentiveNetwork.backward(this.featuresErrorsList)

      val itemsErrors: List<DenseNDArray> = this.recurrentAttentiveNetwork.getInputSequenceErrors()
      val actionEncodingsErrors: List<DenseNDArray> = this.recurrentAttentiveNetwork.getContextLabelsErrors()
      val appliedActions = this.curDecodingContext!!.extendedState.appliedActions

      require(actionEncodingsErrors.size == appliedActions.size - 1) {
        "Errors not aligned with the applied actions. Expected %d, found %d."
          .format(appliedActions.size - 1, actionEncodingsErrors.size)
      }

      appliedActions.zip(actionEncodingsErrors).forEach { (action, errors) -> // the last action is ignored
        this.accumulateActionEncodingErrors(action = action, errors = errors)
      }

      this.accumulateItemsErrors(
        decodingContext = this.curDecodingContext!!,
        itemsErrors = itemsErrors.mapIndexed { itemIndex, errors -> Pair(itemIndex, errors) })
    }
  }

  /**
   * Update the trainable components of this object.
   */
  override fun update() {
    this.actionsVectorsOptimizer.update()
    this.recurrentAttentiveNetwork.update()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.actionsVectorsOptimizer.newBatch()
    this.recurrentAttentiveNetwork.newBatch()
  }

  /**
   * Beat the occurrence of a new example.
   */
  override fun newExample() {
    this.actionsVectorsOptimizer.newExample()
    this.recurrentAttentiveNetwork.newExample()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.actionsVectorsOptimizer.newEpoch()
    this.recurrentAttentiveNetwork.newEpoch()
  }

  /**
   * Check if the backward wasn't called after the application of the last action.
   * It happens when the minimum number of relevant errors is not reached after the application of an action.
   */
  private fun checkMissingBackwardCall() {
    if (this.curDecodingContext!!.extendedState.appliedActions.size > this.featuresErrorsList.size) {
      this.featuresErrorsList.add(this.featuresZerosArray)
    }
  }

  /**
   * @param action an action
   *
   * @return the encoding vector of the given [action]
   */
  private fun getActionEncoding(action: Transition<TransitionType, StateType>.Action): DenseNDArray =
    this.actionsVectorsMap[action.transition.key, action.posTagKey, action.deprelKey]

  /**
   * Accumulate errors of an action embedding.
   *
   * @param action the action
   * @param errors the errors to optimize the embeddings of the given [action]
   */
  private fun accumulateActionEncodingErrors(action: Transition<TransitionType, StateType>.Action,
                                             errors: DenseNDArray) {

    this.actionsVectorsOptimizer.accumulate(
      tId = action.transition.key,
      pId = action.posTagKey,
      dId = action.deprelKey,
      errors = errors
    )
  }

  /**
   * Accumulate the given [itemsErrors] into the related items of the given [decodingContext].
   *
   * @param decodingContext the decoding context that contains extracted features with their errors
   * @param itemsErrors a list of Pairs <itemId, itemErrors>
   */
  private fun accumulateItemsErrors(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    itemsErrors: List<Pair<Int?, DenseNDArray>>) {

    itemsErrors.forEach {
      decodingContext.extendedState.context.accumulateItemErrors(itemIndex = it.first, errors = it.second)
    }
  }
}
