/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.actionsscorer.MPEmbeddingsActionsScorer
import com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure.multiprediction.MPSupportStructure
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionModel
import com.kotlinnlp.simplednn.deeplearning.multipredictionscorer.MultiPredictionOptimizer
import com.kotlinnlp.syntaxdecoder.context.InputContext
import com.kotlinnlp.syntaxdecoder.context.items.StateItem
import com.kotlinnlp.syntaxdecoder.syntax.DependencyRelation
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineState
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.transitions.Shift
import com.kotlinnlp.utils.DictionarySet

/**
 * The ArcEagerSpineActionsScorer.
 *
 * @param network a NeuralNetwork
 * @param optimizer the optimizer of the [network] params
 * @param deprelTags the dictionary set of deprels
 */
open class ArcEagerSpineActionsScorer<
  InputContextType: InputContext<InputContextType, ItemType>,
  ItemType: StateItem<ItemType, *, *>,
  in SupportStructureType : MPSupportStructure>
(
  protected val network: MultiPredictionModel,
  private val optimizer: MultiPredictionOptimizer,
  private val deprelTags: DictionarySet<Deprel>
) :
  MPEmbeddingsActionsScorer<ArcEagerSpineState, ArcEagerSpineTransition, InputContextType, ItemType,
    SupportStructureType>(
    network = network,
    optimizer = optimizer,
    deprelTags = deprelTags
  ) {

  /**
   * Map deprels with their related outcome indices
   */
  private val deprelOutcomeMap: Map<Deprel, Int> = this.buildDeprelOutcomeMap()

  /**
   * The [network] outcome index of this action.
   */
  override val Transition<ArcEagerSpineTransition, ArcEagerSpineState>.Action.outcomeIndex: Int get () = when {
    this.transition is Shift -> 0
    this is DependencyRelation -> this@ArcEagerSpineActionsScorer.deprelOutcomeMap.getValue(this.deprel!!)
    else -> throw RuntimeException("unknown action")
  }

  /**
   * The [network] group id of this transition.
   */
  override val Transition<ArcEagerSpineTransition, ArcEagerSpineState>.groupId: Int get() = Utils.getGroupId(this)

  /**
   * @return a map of deprels to their related outcome indices
   */
  private fun buildDeprelOutcomeMap(): Map<Deprel, Int> {

    val map = mutableMapOf<Deprel, Int>()

    var rootIndex = 0
    var leftIndex = 0
    var rightIndex = 0

    deprelTags.getElements().forEach {

      when(it.direction) {
        Deprel.Position.ROOT -> map[it] = rootIndex++
        Deprel.Position.LEFT -> map[it] = leftIndex++
        Deprel.Position.RIGHT -> map[it] = rightIndex++
        Deprel.Position.NULL -> throw RuntimeException("Deprel direction cannot be NULL")
      }
    }

    return map.toMap()
  }
}
