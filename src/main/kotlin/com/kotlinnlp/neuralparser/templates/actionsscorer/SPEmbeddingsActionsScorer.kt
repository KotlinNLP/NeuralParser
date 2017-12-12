/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.templates.actionsscorer

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensEncodingContext
import com.kotlinnlp.neuralparser.templates.supportstructure.OutputErrorsInit
import com.kotlinnlp.neuralparser.templates.supportstructure.singleprediction.SPSupportStructure
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.neuralparser.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.utils.features.DenseFeaturesErrors
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorer
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorerTrainable
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * The Single Prediction Embeddings ActionsScorer.
 *
 * @param network a NeuralNetwork
 * @param optimizer the optimizer of the [network] params
 * @param deprelTags the dictionary set of deprels
 */
abstract class SPEmbeddingsActionsScorer<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>,
  in SupportStructureType : SPSupportStructure>
(
  private val network: NeuralNetwork,
  private val optimizer: ParamsOptimizer<NetworkParameters>,
  private val deprelTags: DictionarySet<Deprel>
) :
  ActionsScorerTrainable<
    StateType,
    TransitionType,
    InputContextType,
    DenseItem,
    DenseFeaturesErrors,
    DenseFeatures,
    SupportStructureType>()
{

  /**
   * The [network] outcome index of this action.
   */
  abstract protected val Transition<TransitionType, StateType>.Action.outcomeIndex: Int

  /**
   * Assign scores to the actions contained into the given [decodingContext] using the features contained in it and the
   * given [supportStructure].
   *
   * @param decodingContext the decoding context that contains the actions to score
   * @param supportStructure the decoding support structure
   */
  override fun score(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: SupportStructureType) {

    val prediction: DenseNDArray = supportStructure.processor.forward(featuresArray = decodingContext.features.array)

    decodingContext.actions.forEach { action -> action.score = prediction[action.outcomeIndex] }
  }

  /**
   * Backward errors through this [ActionsScorer], starting from the scored actions in the given [decodingContext].
   * Errors are required to be already set into the output actions properly.
   *
   * @param decodingContext the decoding context that contains the scored actions
   * @param supportStructure the decoding support structure
   * @param propagateToInput a Boolean indicating whether errors must be propagated to the input items
   */
  override fun backward(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: SupportStructureType,
    propagateToInput: Boolean) {

    supportStructure.processor.backward(
      outputErrors = this.getOutputErrors(
        actions = decodingContext.actions,
        outcomes = supportStructure.processor.structure.outputLayer.outputArray.values,
        initType = supportStructure.outputErrorsInit),
      propagateToInput = propagateToInput)

    this.optimizer.accumulate(supportStructure.processor.getParamsErrors(copy = true))
  }

  /**
   * @param decodingContext the decoding context that contains the scored actions
   * @param supportStructure the decoding support structure
   *
   * @return the errors of the features used to score the actions in the given [decodingContext]
   */
  override fun getFeaturesErrors(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: SupportStructureType
  ) = DenseFeaturesErrors(supportStructure.processor.getInputErrors(copy = true))

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.optimizer.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.optimizer.newExample()
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.optimizer.newEpoch()
  }

  /**
   * Update the scorer.
   */
  override fun update() {
    this.optimizer.update()
  }

  /**
   * Link [actions] errors to the output errors array of the neural network used to score the given [actions].
   *
   * @param actions the list of scored actions (containing their errors)
   * @param outcomes the array of the network outcomes related to the scoring of the [actions]
   * @param initType the type of errors initialization
   *
   * @return the array of output errors
   */
  private fun getOutputErrors(actions: List<Transition<TransitionType, StateType>.Action>,
                              outcomes: DenseNDArray,
                              initType: OutputErrorsInit): DenseNDArray {

    val errors: DenseNDArray = when (initType) {
      OutputErrorsInit.AllZeros -> DenseNDArrayFactory.zeros(outcomes.shape)
      OutputErrorsInit.AllErrors -> outcomes.copy()
    }

    actions.forEach { errors[it.outcomeIndex] = it.error }

    return errors
  }
}
