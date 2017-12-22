/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint2

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint2.actionsembeddings.ActionsVectorsMap
import com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint2.actionsembeddings.ActionsVectorsOptimizer
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.templates.featuresextractor.EmbeddingsFeaturesExtractor
import com.kotlinnlp.neuralparser.templates.featuresextractor.TokensWindowFeaturesExtractorTrainable
import com.kotlinnlp.neuralparser.templates.supportstructure.compositeprediction.ATPDJointSupportStructure
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.FeaturesExtractor
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.ArcStandardTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Root
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcstandard.transitions.Shift
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.StateView
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext
import com.kotlinnlp.syntaxdecoder.utils.getItemOrNull

/**
 * The [EmbeddingsFeaturesExtractor] that extracts features combining word Embeddings and ambiguous POS vectors for the
 * ArcStandard transition system.
 *
 * @param actionsVectors the encoding vectors of the actions
 * @param actionsVectorsOptimizer the optimizer of the [actionsVectors]
 * @param actionsEncoderOptimizer the optimizer of the RNN used to encode actions
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcStandardATPDFeaturesExtractor(
  private val actionsVectors: ActionsVectorsMap,
  private val actionsVectorsOptimizer: ActionsVectorsOptimizer,
  private val actionsEncoderOptimizer: ParamsOptimizer<NetworkParameters>,
  private val deprelTags: DictionarySet<Deprel>,
  private val posTags: DictionarySet<POSTag>,
  private val actionsEncodingSize: Int
) : TokensWindowFeaturesExtractorTrainable<
  StackBufferState,
  ArcStandardTransition,
  TokensAmbiguousPOSContext,
  DenseFeatures,
  StateView<StackBufferState>,
  ATPDJointSupportStructure>() {

  /**
   * The actions embeddings map key of the transition of this action.
   */
  private val Transition<ArcStandardTransition, StackBufferState>.key: Int
    get() = when (this) {
      is Shift -> 0
      is Root -> 1
      is ArcLeft -> 2
      is ArcRight -> 3
      else -> throw RuntimeException("unknown transition")
    }

  /**
   * The actions embeddings map key of the deprel of this action.
   */
  private val Transition<ArcStandardTransition, StackBufferState>.Action.deprelKey: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardATPDFeaturesExtractor.deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The actions embeddings map key of the POS tag of this action.
   */
  private val Transition<ArcStandardTransition, StackBufferState>.Action.posTagKey: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> this@ArcStandardATPDFeaturesExtractor.posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The list of actions applied to the current state.
   */
  private val appliedActions = mutableListOf<Transition<ArcStandardTransition, StackBufferState>.Action?>()

  /**
   * The list of encoding vectors errors of the actions applied to the current state.
   */
  private val appliedActionsEncodingErrors = mutableListOf<DenseNDArray>()

  /**
   *
   */
  private val appliedActionZerosErrors = DenseNDArrayFactory.zeros(Shape(this.actionsEncodingSize))

  /**
   * The RNN used to encode the actions.
   */
  private lateinit var actionsEncoder: RecurrentNeuralProcessor<DenseNDArray>

  /**
   * Extract features using the given [decodingContext] amd [supportStructure].
   *
   * @param decodingContext the decoding context
   * @param supportStructure the decoding support structure
   *
   * @return the extracted features
   */
  override fun extract(
    decodingContext: DecodingContext<StackBufferState, ArcStandardTransition, TokensAmbiguousPOSContext, DenseItem, DenseFeatures>,
    supportStructure: ATPDJointSupportStructure
  ): DenseFeatures {

    val trainingMode = decodingContext.extendedState.context.trainingMode
    val appliedActions = decodingContext.extendedState.appliedActions

    if (trainingMode && appliedActions.isEmpty()){
      this.appliedActionsEncodingErrors.clear()
      this.appliedActions.clear()
    }

    if (trainingMode && this.appliedActions.size > this.appliedActionsEncodingErrors.size) {
      // The backward wasn't called for the last applied action
      this.appliedActionsEncodingErrors.add(this.appliedActionZerosErrors)
    }

    val lastAppliedActionVector: DenseNDArray = if (appliedActions.isEmpty()) {

      if (trainingMode) {

        if (this.appliedActions.isNotEmpty()) this.backwardActionsEncoding()

        this.appliedActions.add(null)
      }

      this.actionsVectors.getNullVector()

    } else {
      val action = appliedActions.last()
      if (trainingMode) this.appliedActions.add(action)
      this.actionsVectors[action.transition.key, action.posTagKey, action.deprelKey]
    }

    this.actionsEncoder = supportStructure.actionProcessor

    return DenseFeatures(array = concatVectorsV(
      supportStructure.actionProcessor.forward(
        featuresArray = lastAppliedActionVector,
        firstState = appliedActions.isEmpty()),
      this.extractViewFeatures(
        stateView = StateView(state = decodingContext.extendedState.state),
        context = decodingContext.extendedState.context))
    )
  }

  /**
   * @param tokenId a token id
   * @param context a tokens dense context
   *
   * @return the token representation as dense array
   */
  override fun getTokenEncoding(tokenId: Int?, context: TokensAmbiguousPOSContext): DenseNDArray =
    context.getTokenEncoding(tokenId)

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
    decodingContext: DecodingContext<StackBufferState, ArcStandardTransition, TokensAmbiguousPOSContext, DenseItem,
      DenseFeatures>,
    supportStructure: ATPDJointSupportStructure,
    propagateToInput: Boolean
  ) {

    if (propagateToInput) {

      val itemsWindow: List<Int?> = this.getTokensWindow(StateView(state = decodingContext.extendedState.state))

      val errors: DenseNDArray = decodingContext.features.errors.array
      val splitErrors: Array<DenseNDArray> = errors.splitV(
        this.actionsEncodingSize,
        errors.length - this.actionsEncodingSize
      )

      this.appliedActionsEncodingErrors.add(splitErrors[0])
      val tokensErrors: Array<DenseNDArray> = splitErrors[1].splitV(decodingContext.extendedState.context.encodingSize)

      this.accumulateItemsErrors(decodingContext = decodingContext, itemsErrors = itemsWindow.zip(tokensErrors))
    }
  }

  /**
   * Beat the occurrence of a new example.
   */
  override fun newExample() {
    this.actionsEncoderOptimizer.newExample()
  }

  /**
   * Beat the occurrence of a new batch.
   */
  override fun newBatch() {
    this.actionsEncoderOptimizer.newBatch()
  }

  /**
   * Beat the occurrence of a new epoch.
   */
  override fun newEpoch() {
    this.actionsEncoderOptimizer.newEpoch()
  }

  /**
   * Update the trainable components of this [FeaturesExtractor].
   */
  override fun update() {

    if (this.appliedActions.isNotEmpty()) this.backwardActionsEncoding()

    this.actionsVectorsOptimizer.update()
    this.actionsEncoderOptimizer.update()
  }

  /**
   * Get the tokens window respect to a given state
   *
   * @param stateView a view of the state
   *
   * @return the tokens window as list of Int
   */
  override fun getTokensWindow(stateView: StateView<StackBufferState>) = listOf(
    stateView.state.stack.getItemOrNull(-3),
    stateView.state.stack.getItemOrNull(-2),
    stateView.state.stack.getItemOrNull(-1),
    stateView.state.buffer.getItemOrNull(0))

  /**
   *
   */
  private fun backwardActionsEncoding() {

    if (this.appliedActions.size > this.appliedActionsEncodingErrors.size) {
      // The backward wasn't called for the last applied action
      this.appliedActionsEncodingErrors.add(this.appliedActionZerosErrors)
    }

    this.actionsEncoder.backward(
      outputErrorsSequence = this.appliedActionsEncodingErrors.toTypedArray(),
      propagateToInput = true)

    this.actionsEncoderOptimizer.accumulate(this.actionsEncoder.getParamsErrors(copy = false))

    this.actionsEncoder.getInputSequenceErrors(copy = false).forEachIndexed { i, errors ->
      this.accumulateActionEmbeddingErrors(errors = errors, actionIndex = i)
    }

    this.appliedActions.clear()
    this.appliedActionsEncodingErrors.clear()
  }

  /**
   *
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
   * Accumulate the given [itemsErrors] into the related items of the given [decodingContext].
   *
   * @param decodingContext the decoding context that contains extracted features with their errors
   * @param itemsErrors a list of Pairs <itemId, itemErrors>
   */
  private fun accumulateItemsErrors(
    decodingContext: DecodingContext<StackBufferState, ArcStandardTransition, TokensAmbiguousPOSContext, DenseItem, DenseFeatures>,
    itemsErrors: List<Pair<Int?, DenseNDArray>>) {

    itemsErrors.forEach {
      decodingContext.extendedState.context.accumulateItemErrors(itemIndex = it.first, errors = it.second)
    }
  }
}
