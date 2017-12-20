/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint.actionsembeddings

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsOptimizer
import com.kotlinnlp.simplednn.utils.MultiMap
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The optimizer of the [ActionsEmbeddingsMap].
 *
 * @param actionsEmbeddingsMap the [ActionsEmbeddingsMap] to optimize
 * @param updateMethod the update method for the optimization (e.g. ADAM, AdaGrad, ...)
 */
class ActionsEmbeddingsOptimizer(actionsEmbeddingsMap: ActionsEmbeddingsMap, updateMethod: UpdateMethod<*>) {

  /**
   * The optimizer of the 'nullEmbedding' map.
   */
  private val nullEmbeddingOptimizer = EmbeddingsOptimizer(
    embeddingsMap = actionsEmbeddingsMap.nullEmbeddingMap,
    updateMethod = updateMethod)

  /**
   * A nested multimap that maps transition and POS tag ids to the optimizers of the nested deprel embeddings maps.
   */
  private val tpdEmbeddingOptimizer: MultiMap<EmbeddingsOptimizer<Int>> = MultiMap(
    mapOf(*(0 until actionsEmbeddingsMap.transitionsSize).map { tId ->
      // transition level
      Pair(
        tId,
        mapOf(*(0 until actionsEmbeddingsMap.posTagsSize).map { pId ->
          // POS tag level
          Pair(
            pId,
            EmbeddingsOptimizer( // deprel level
              embeddingsMap = actionsEmbeddingsMap[tId, pId],
              updateMethod = updateMethod)
          )
        }.toTypedArray())
      )
    }.toTypedArray())
  )

  /**
   * Update the embeddings.
   */
  fun update() {

    this.nullEmbeddingOptimizer.update()
    this.tpdEmbeddingOptimizer.forEach { _, _, embeddingsMap -> embeddingsMap.update() }
  }

  /**
   * Accumulate errors of the embedding vector associated to the given ids.
   *
   * @param tId the transition id
   * @param pId the POS tag id
   * @param dId the deprel id
   * @param errors errors to accumulate
   */
  fun accumulate(tId: Int, pId: Int, dId: Int, errors: DenseNDArray) {
    this.tpdEmbeddingOptimizer[tId, pId]!!.accumulate(embeddingKey = dId, errors = errors)
  }

  /**
   * Accumulate errors of the 'nullEmbedding'.
   *
   * @param errors errors to accumulate
   */
  fun accumulateNullEmbeddingsErrors(errors: DenseNDArray) {
    this.nullEmbeddingOptimizer.accumulate(embeddingKey = null, errors = errors)
  }
}
