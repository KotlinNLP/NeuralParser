/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arcrelocate.atpdjoint

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.actionsscorer.TPDJointEmbeddingsActionsScorer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcrelocate.ArcRelocateTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.templates.StackBufferState
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.simplednn.core.neuralnetwork.NetworkParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.deeplearning.multitasknetwork.MultiTaskNetworkParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arcrelocate.transitions.*
import com.kotlinnlp.utils.DictionarySet

/**
 * The ArcRelocateTPDJointActionsScorer.
 *
 * @param activationFunction the function used to activate the actions scores
 * @param transitionOptimizer the optimizer of the transition network params
 * @param posDeprelOptimizer the optimizer of the POS-deprel network params
 * @param posTags the dictionary set of POS tags
 * @param deprelTags the dictionary set of deprels
 */
class ArcRelocateTPDJointActionsScorer(
  activationFunction: ActivationFunction?,
  transitionOptimizer: ParamsOptimizer<NetworkParameters>,
  posDeprelOptimizer: ParamsOptimizer<MultiTaskNetworkParameters>,
  private val deprelTags: DictionarySet<Deprel>,
  private val posTags: DictionarySet<POSTag>
) : TPDJointEmbeddingsActionsScorer<StackBufferState, ArcRelocateTransition, TokensAmbiguousPOSContext>(
  activationFunction = activationFunction,
  transitionOptimizer = transitionOptimizer,
  posDeprelOptimizer = posDeprelOptimizer) {

  /**
   * The transition network outcome index of this transition.
   */
  override val Transition<ArcRelocateTransition, StackBufferState>.outcomeIndex: Int
    get() = when (this) {
      is Shift -> 0
      is Root -> 1
      is ArcLeft -> 2
      is ArcRight -> 3
      is Wait -> 4
      else -> throw RuntimeException("unknown transition")
    }

  /**
   * The deprel network outcome index of this action.
   */
  override val Transition<ArcRelocateTransition, StackBufferState>.Action.deprelOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> deprelTags.getId(this.deprel!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }

  /**
   * The pos network outcome index of this action.
   */
  override val Transition<ArcRelocateTransition, StackBufferState>.Action.posTagOutcomeIndex: Int
    get() = when {
      this.transition is Shift -> 0
      this is DependencyRelation -> posTags.getId(this.posTag!!)!! + 1 // + shift offset
      else -> throw RuntimeException("unknown action")
    }
  /**
   * Combine the TPD scores to obtain the actions scores.
   *
   * @param actions the list of actions to score
   * @param transitionPrediction the outcome array of the transition network prediction
   * @param posPrediction the outcome array of the POS tag network prediction
   * @param deprelPrediction the outcome array of the deprel network prediction
   *
   * @return the array of action scores
   */
  override fun combineScores(actions: List<Transition<ArcRelocateTransition, StackBufferState>.Action>,
                             transitionPrediction: DenseNDArray,
                             posPrediction: DenseNDArray,
                             deprelPrediction: DenseNDArray): DenseNDArray =
    DenseNDArrayFactory.arrayOf(doubleArrayOf(*DoubleArray(
      size = actions.size,
      init = { i ->
        val action = actions[i]

        if (action.transition is Wait) {

          transitionPrediction[action.transition.outcomeIndex]

        } else {

          val t = transitionPrediction[action.transition.outcomeIndex]
          val p = posPrediction[action.posTagOutcomeIndex]
          val d = deprelPrediction[action.deprelOutcomeIndex]

          t + p + d
        }
      }
    )))

  /**
   * Link [actions] errors to the output errors array of the neural network used to score the given [actions].
   *
   * @param actions the list of scored actions (containing their errors)
   * @param transitionOutcomes the array of the network outcomes related to the scoring of the transitions
   * @param deprelOutcomes the array of the network outcomes related to the scoring of the deprels
   * @param posOutcomes the array of the network outcomes related to the scoring of the POS tags
   *
   * @return the array of output errors
   */
  override fun getOutputErrors(actions: List<Transition<ArcRelocateTransition, StackBufferState>.Action>,
                               transitionOutcomes: DenseNDArray,
                               posOutcomes: DenseNDArray,
                               deprelOutcomes: DenseNDArray): Triple<DenseNDArray, DenseNDArray, DenseNDArray> {

    val transitionErrors = DenseNDArrayFactory.zeros(transitionOutcomes.shape)
    val posErrors = DenseNDArrayFactory.zeros(posOutcomes.shape)
    val deprelErrors = DenseNDArrayFactory.zeros(deprelOutcomes.shape)

    actions.forEach { action ->
      transitionErrors[action.transition.outcomeIndex] += action.error

      if (action.transition !is Wait) {
        posErrors[action.posTagOutcomeIndex] += action.error
        deprelErrors[action.deprelOutcomeIndex] += action.error
      }
    }

    return Triple(transitionErrors, posErrors, deprelErrors)
  }
}
