/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.actionsscorer

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.multiprediction.MPSupportStructure
import com.kotlinnlp.simplednn.utils.DictionarySet
import com.kotlinnlp.neuralparser.utils.features.GroupedDenseFeatures
import com.kotlinnlp.neuralparser.utils.features.GroupedDenseFeaturesErrors
import com.kotlinnlp.simplednn.utils.MultiMap
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionModel
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.context.InputContext
import com.kotlinnlp.syntaxdecoder.context.items.StateItem
import com.kotlinnlp.syntaxdecoder.utils.DecodingContext
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorer
import com.kotlinnlp.syntaxdecoder.modules.actionsscorer.ActionsScorerTrainable
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * The Multi Prediction Embeddings ActionsScorer.
 *
 * @param network a NeuralNetwork
 * @param optimizer the optimizer of the [network] params
 * @param deprelTags the dictionary set of deprels
 */
abstract class MPEmbeddingsActionsScorer<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  InputContextType: InputContext<InputContextType, ItemType>,
  ItemType: StateItem<ItemType, *, *>,
  in SupportStructureType : MPSupportStructure>
(
  private val network: MultiPredictionModel,
  private val optimizer: MultiPredictionOptimizer,
  private val deprelTags: DictionarySet<Deprel>
) :
  ActionsScorerTrainable<
    StateType,
    TransitionType,
    InputContextType,
    ItemType,
    GroupedDenseFeaturesErrors,
    GroupedDenseFeatures,
    SupportStructureType>()
{

  /**
   * Assign scores to the actions contained into the given [decodingContext] using the features contained in it and the
   * given [supportStructure].
   *
   * @param decodingContext the decoding context that contains the actions to score
   * @param supportStructure the decoding support structure
   */
  override fun score(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, ItemType, GroupedDenseFeatures>,
    supportStructure: SupportStructureType) {

    val prediction: MultiMap<DenseNDArray> = supportStructure.scorer.score(decodingContext.features.featuresMap)

    decodingContext.actions
      .filter { it.isAllowed }
      .forEach { action ->
      action.score = prediction.getValue(i = action.transition.groupId, j = action.transition.id)[action.outcomeIndex]
    }
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
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, ItemType, GroupedDenseFeatures>,
    supportStructure: SupportStructureType,
    propagateToInput: Boolean) {

    supportStructure.scorer.backward(
      outputsErrors = this.getErrorsMultimap(decodingContext.actions),
      propagateToInput = propagateToInput)

    supportStructure.scorer.getParamsErrors(copy = false).forEach { groupId, _, errors ->
      this.optimizer.accumulate(networkIndex = groupId, errors = errors)
    }
  }

  /**
   * @param decodingContext the decoding context that contains the scored actions
   * @param supportStructure the decoding support structure
   *
   * @return the errors of the features used to score the actions in the given [decodingContext]
   */
  override fun getFeaturesErrors(
    decodingContext: DecodingContext<StateType, TransitionType, InputContextType, ItemType, GroupedDenseFeatures>,
    supportStructure: SupportStructureType
  ) = GroupedDenseFeaturesErrors(errorsMap = supportStructure.scorer.getInputErrors(copy = false))

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {
    this.optimizer.newEpoch()
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
   * The [network] group id of this transition.
   */
  abstract protected val Transition<TransitionType, StateType>.Action.outcomeIndex: Int

  /**
   * The group id of this transition.
   */
  protected abstract val Transition<TransitionType, StateType>.groupId: Int

  /**
   * Link [actions] errors to the output errors array of the multi-prediction scorer used to score the given [actions].
   * Zeros errors are given to outcomes that wasn't linked to a generated action (actions not possible).
   *
   * @param actions the list of scored actions (containing their errors)
   *
   * @return the multimap of output errors related to the actions errors, as they have been scored
   */
  private fun getErrorsMultimap(actions: List<Transition<TransitionType, StateType>.Action>): MultiMap<DenseNDArray> {

    val errorsMap = mutableMapOf<Int, MutableMap<Int, DenseNDArray>>()

    actions
      .filter { it.isAllowed }
      .groupBy { it.transition.groupId }
      .forEach { groupId, actionsGroup ->

        actionsGroup
          .groupBy { it.transition.id }
          .forEach { transitionId, actions ->

            if (actions.any { it.error != 0.0}) {

              val errors: DenseNDArray = DenseNDArrayFactory.zeros(shape = Shape(
                this@MPEmbeddingsActionsScorer.network.networks[groupId].layersConfiguration.last().size))

              actions
                .filter { it.error != 0.0 }
                .forEach { errors[it.outcomeIndex] = it.error }

              if (groupId !in errorsMap) {
                errorsMap[groupId] = mutableMapOf()
              }

              errorsMap[groupId]!![transitionId] = errors
            }
          }
      }

    return MultiMap(data = errorsMap)
  }
}
