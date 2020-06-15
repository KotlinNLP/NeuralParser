/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.LossCriterion
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.LossCriterionType
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The model of the [Labeler].
 *
 * @property contextEncodingSize the size of the token encoding vectors
 * @property grammaticalConfigurations the dictionary set of all the possible grammatical configurations
 * @property lossCriterionType the training mode
 */
class LabelerModel(
  val contextEncodingSize: Int,
  val grammaticalConfigurations: DictionarySet<GrammaticalConfiguration>,
  val lossCriterionType: LossCriterionType
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The score threshold above which to consider a labeler output valid.
   * It makes sense with the Softmax activation function.
   */
  internal val labelerScoreThreshold: Double = 1.0 / this.grammaticalConfigurations.size

  /**
   * The Network model that predicts the grammatical configurations.
   */
  val networkModel = StackedLayersParameters(
    LayerInterface(sizes = listOf(this.contextEncodingSize, this.contextEncodingSize)),
    LayerInterface(
      size = this.contextEncodingSize,
      connectionType = LayerType.Connection.Affine,
      activationFunction = Tanh),
    LayerInterface(
      type = LayerType.Input.Dense,
      size = this.grammaticalConfigurations.size,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = when (this.lossCriterionType) {
        LossCriterionType.Softmax -> Softmax()
        LossCriterionType.HingeLoss -> null
      })
  )

  /**
   * Return the errors of a given labeler predictions, respect to a gold dependency tree.
   * Errors are calculated comparing the last predictions done with the given gold grammatical configurations.
   *
   * @param predictions the current network predictions
   * @param goldTree the gold tree of the sentence
   *
   * @return a list of predictions errors
   */
  fun calculateLoss(predictions: List<DenseNDArray>, goldTree: DependencyTree.Labeled): List<DenseNDArray> {

    val errorsList = mutableListOf<DenseNDArray>()

    predictions.forEachIndexed { tokenIndex, prediction ->

      val tokenId: Int = goldTree.elements[tokenIndex]
      val errors: DenseNDArray = LossCriterion(this.lossCriterionType).getPredictionErrors(
        prediction = prediction,
        goldIndex = this.grammaticalConfigurations.getId(goldTree.getConfiguration(tokenId))!!)

      errorsList.add(errors)
    }

    return errorsList
  }
}
