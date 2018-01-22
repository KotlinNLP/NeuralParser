/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.actionsembeddings

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.core.optimizer.ScheduledUpdater
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.utils.scheduling.BatchScheduling
import com.kotlinnlp.simplednn.utils.scheduling.EpochScheduling
import com.kotlinnlp.simplednn.utils.scheduling.ExampleScheduling

/**
 * The optimizer of the [ActionsVectorsMap].
 *
 * @param actionsVectorsMap the [ActionsVectorsMap] to optimize
 * @param updateMethod the update method for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class ActionsVectorsOptimizer(
  actionsVectorsMap: ActionsVectorsMap,
  val updateMethod: UpdateMethod<*>
) : ScheduledUpdater {

  /**
   * The optimizer of the transition embeddings maps.
   */
  private val transitionEmbeddingOptimizer: EmbeddingsOptimizer<Int> = EmbeddingsOptimizer(
    embeddingsMap = actionsVectorsMap.transitionEmbeddings,
    updateMethod = updateMethod)

  /**
   * The optimizer of the posTag embeddings maps.
   */
  private val posTagEmbeddingOptimizer: EmbeddingsOptimizer<Int> = EmbeddingsOptimizer(
    embeddingsMap = actionsVectorsMap.posTagEmbeddings,
    updateMethod = updateMethod)

  /**
   * The optimizer of the deprel embeddings maps.
   */
  private val deprelEmbeddingOptimizer: EmbeddingsOptimizer<Int> = EmbeddingsOptimizer(
    embeddingsMap = actionsVectorsMap.deprelEmbeddings,
    updateMethod = updateMethod)

  /**
   * Update the embeddings.
   */
  override fun update() {

    this.transitionEmbeddingOptimizer.update()
    this.posTagEmbeddingOptimizer.update()
    this.deprelEmbeddingOptimizer.update()
  }

  /**
   * Accumulate errors of the embedding vectors associated to the given ids.
   *
   * @param tId the transition id
   * @param pId the POS tag id (can be null)
   * @param dId the deprel id (can be null)
   * @param errors errors to accumulate
   */
  fun accumulate(tId: Int, pId: Int?, dId: Int?, errors: DenseNDArray) {

    val splitErrors: Array<DenseNDArray> = errors.splitV(errors.length / 3)

    this.transitionEmbeddingOptimizer.accumulate(embeddingKey = tId, errors = splitErrors[0])
    this.posTagEmbeddingOptimizer.accumulate(embeddingKey = pId, errors = splitErrors[1])
    this.deprelEmbeddingOptimizer.accumulate(embeddingKey = dId, errors = splitErrors[2])
  }

  /**
   * Accumulate errors of the 'nullEmbedding'.
   *
   * @param errors errors to accumulate
   */
  fun accumulateNullEmbeddingsErrors(errors: DenseNDArray) {

    val splitErrors: Array<DenseNDArray> = errors.splitV(errors.length / 3)

    this.transitionEmbeddingOptimizer.accumulate(embeddingKey = null, errors = splitErrors[0])
    this.posTagEmbeddingOptimizer.accumulate(embeddingKey = null, errors = splitErrors[1])
    this.deprelEmbeddingOptimizer.accumulate(embeddingKey = null, errors = splitErrors[2])
  }

  /**
   * Method to call every new epoch.
   */
  override fun newEpoch() {

    if (this.updateMethod is EpochScheduling) {
      this.updateMethod.newEpoch()
    }
  }

  /**
   * Method to call every new batch.
   */
  override fun newBatch() {

    if (this.updateMethod is BatchScheduling) {
      this.updateMethod.newBatch()
    }
  }

  /**
   * Method to call every new example.
   */
  override fun newExample() {

    if (this.updateMethod is ExampleScheduling) {
      this.updateMethod.newExample()
    }
  }
}
