/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.actionsscorer

import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensEncodingContext
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.compositeprediction.TDSupportStructure
import com.kotlinnlp.neuralparser.parsers.transitionbased.utils.features.DenseFeatures
import com.kotlinnlp.neuralparser.parsers.transitionbased.utils.features.DenseFeaturesErrors
import com.kotlinnlp.neuralparser.parsers.transitionbased.utils.items.DenseItem
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorer
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorerTrainable
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * The Transition+Deprel Embeddings ActionsScorer.
 *
 * @param transitionOptimizer the optimizer of the transition network params
 * @param deprelOptimizer the optimizer of the deprel network params
 */
abstract class TDEmbeddingsActionsScorer<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: TokensEncodingContext<InputContextType>>
(
  private val transitionOptimizer: ParamsOptimizer<NetworkParameters>,
  private val deprelOptimizer: ParamsOptimizer<NetworkParameters>
) : ActionsScorerTrainable<
  StateType,
  TransitionType,
  InputContextType,
  DenseItem,
  DenseFeaturesErrors,
  DenseFeatures,
  TDSupportStructure>() {

  /**
   * The deprel network outcome index of this action.
   */
  protected abstract val Transition<TransitionType, StateType>.Action.outcomeIndex: Int

  /**
   * The transition network outcome index of this transition.
   */
  protected abstract val Transition<TransitionType, StateType>.outcomeIndex: Int

  /**
   * Assign scores to the actions contained into the given [decodingContext] using the features contained in it and the
   * given [supportStructure].
   *
   * @param decodingContext the decoding context that contains the actions to score
   * @param supportStructure the decoding support structure
   */
  override fun score(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: TDSupportStructure) {

    val deprelPrediction: DenseNDArray = supportStructure.deprelProcessor.forward(decodingContext.features.array)
    val transitionPrediction: DenseNDArray =
      supportStructure.transitionProcessor.forward(decodingContext.features.array)

    decodingContext.actions.forEach { action ->
      action.score = deprelPrediction[action.outcomeIndex] + transitionPrediction[action.transition.outcomeIndex]
    }
  }

  /**
   * Backward errors through this [ActionsScorer], starting from the scored actions in the given [decodingContext].
   * Errors are required to be already set into the output actions properly.
   *
   * @param decodingContext the decoding context that contains the scored actions
   * @param supportStructure the decoding support structure
   */
  override fun backward(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: TDSupportStructure) {

    val (deprelErrors, transitionErrors) = this.getOutputErrors(
      actions = decodingContext.actions,
      deprelOutcomes = supportStructure.deprelProcessor.structure.outputLayer.outputArray.values,
      transitionOutcomes = supportStructure.transitionProcessor.structure.outputLayer.outputArray.values)

    supportStructure.deprelProcessor.backward(deprelErrors)
    supportStructure.transitionProcessor.backward(transitionErrors)

    this.deprelOptimizer.accumulate(supportStructure.deprelProcessor.getParamsErrors(copy = true))
    this.transitionOptimizer.accumulate(supportStructure.transitionProcessor.getParamsErrors(copy = true))
  }

  /**
   * @param decodingContext the decoding context that contains the scored actions
   * @param supportStructure the decoding support structure
   *
   * @return the errors of the features used to score the actions in the given [decodingContext]
   */
  override fun getFeaturesErrors(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, DenseItem, DenseFeatures>,
    supportStructure: TDSupportStructure
  ) = DenseFeaturesErrors(
    supportStructure.deprelProcessor.getInputErrors(copy = true)
      .sum(supportStructure.transitionProcessor.getInputErrors(copy = true))
  )

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.transitionOptimizer.newBatch()
    this.deprelOptimizer.newBatch()
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {
    this.transitionOptimizer.newExample()
    this.deprelOptimizer.newExample()
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {
    this.transitionOptimizer.newEpoch()
    this.deprelOptimizer.newEpoch()
  }

  /**
   * Update the scorer.
   */
  override fun update() {
    this.transitionOptimizer.update()
    this.deprelOptimizer.update()
  }

  /**
   * Link [actions] errors to the output errors array of the neural network used to score the given [actions].
   *
   * @param actions the list of scored actions (containing their errors)
   * @param deprelOutcomes the array of the network outcomes related to the scoring of the deprels
   * @param transitionOutcomes the array of the network outcomes related to the scoring of the transitions
   *
   * @return the array of output errors
   */
  private fun getOutputErrors(actions: List<Transition<TransitionType, StateType>.Action>,
                              deprelOutcomes: DenseNDArray,
                              transitionOutcomes: DenseNDArray): Pair<DenseNDArray, DenseNDArray> {

    val deprelErrors = DenseNDArrayFactory.zeros(deprelOutcomes.shape)
    val transitionErrors = DenseNDArrayFactory.zeros(transitionOutcomes.shape)

    actions.forEach { action ->
      deprelErrors[action.outcomeIndex] = action.error
      transitionErrors[action.transition.outcomeIndex] += action.error
    }

    return Pair(deprelErrors, transitionErrors)
  }
}
