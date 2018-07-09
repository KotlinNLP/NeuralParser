/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler

import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils.LossCriterion
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils.LossCriterionType
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.neuralnetwork.NeuralNetwork
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The model of the [DeprelLabeler].
 *
 * @property contextEncodingSize the size of the token encoding vectors
 * @property deprels the dictionary set of all possible deprels
 * @property lossCriterionType the training mode
 */
class DeprelLabelerModel(
  val contextEncodingSize: Int,
  val deprels: DictionarySet<Deprel>,
  val lossCriterionType: LossCriterionType
) : Serializable {

  /**
   * The Network model that predicts the Deprels
   */
  val networkModel: NeuralNetwork = NeuralNetwork(
    LayerInterface(sizes = listOf(this.contextEncodingSize, this.contextEncodingSize)),
    LayerInterface(
      size = this.contextEncodingSize,
      connectionType = LayerType.Connection.Affine,
      activationFunction = Tanh()),
    LayerInterface(
      type = LayerType.Input.Dense,
      size = this.deprels.size,
      dropout = 0.0,
      connectionType = LayerType.Connection.Feedforward,
      activationFunction = when (this.lossCriterionType) {
        LossCriterionType.Softmax -> Softmax()
        LossCriterionType.HingeLoss -> null
      })
  )

  /**
   * Return the errors of a given labeler predictions, respect to a gold dependency tree.
   * Errors are calculated comparing the last predictions done with the given gold deprels.
   *
   * @param predictions the labeler predictions
   * @param goldDeprels the list of gold deprels
   *
   * @return a list of predictions errors
   */
  fun calculateLoss(predictions: List<DeprelLabeler.Prediction>, goldDeprels: Array<Deprel?>): List<DenseNDArray> {

    val errorsList = mutableListOf<DenseNDArray>()

    predictions.forEachIndexed { tokenId, prediction ->

      val goldDeprel: Deprel = goldDeprels[tokenId]!!
      val goldDeprelIndex: Int = this.deprels.getId(goldDeprel)!!

      errorsList.add(LossCriterion(this.lossCriterionType).getPredictionErrors(
        prediction = prediction.deprels, goldIndex = goldDeprelIndex))
    }

    return errorsList
  }
}