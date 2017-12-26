/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils.actionsembeddings

import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import java.io.Serializable

/**
 * The container of vectors that encode actions, mapped to their TPD ids.
 *
 * @property size the size of each embedding vector
 * @property transitionsSize the number of transitions
 * @property posTagsSize the number of POS tags
 * @property deprelsSize the number of deprels
 */
class ActionsVectorsMap(
  val size: Int,
  val transitionsSize: Int,
  val posTagsSize: Int,
  val deprelsSize: Int
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed from Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * An Embeddings map that maps transition ids to embeddings vectors.
   */
  val transitionEmbeddings = EmbeddingsMap<Int>(size = this.size)

  /**
   * An Embeddings map that maps POS tag ids to embeddings vectors.
   */
  val posTagEmbeddings = EmbeddingsMap<Int>(size = this.size)

  /**
   * An Embeddings map that maps deprel ids to embeddings vectors.
   */
  val deprelEmbeddings = EmbeddingsMap<Int>(size = this.size)

  /**
   * Check sizes and initialize the deprels embeddings maps.
   */
  init {
    require(this.size > 0) { "The embeddings vectors size must be > 0" }
    require(this.transitionsSize > 0) { "The transitions size must be > 0" }
    require(this.posTagsSize > 0) { "The POS tags size must be > 0" }
    require(this.deprelsSize > 0) { "The deprels size must be > 0" }

    (0 until this.transitionsSize).forEach { dId -> this.transitionEmbeddings.set(dId)  }
    (0 until this.deprelsSize).forEach { dId -> this.deprelEmbeddings.set(dId)  }
    (0 until this.posTagsSize).forEach { dId -> this.posTagEmbeddings.set(dId)  }
  }

  /**
   * @param tId the transition id
   * @param pId the POS tag id
   * @param dId the deprel id
   *
   * @throws KotlinNullPointerException if [tId] is not within the range [0, [transitionsSize])
   * @throws KotlinNullPointerException if [pId] is not within the range [0, [posTagsSize])
   * @throws KotlinNullPointerException if [dId] is not within the range [0, [deprelsSize])
   *
   * @return the embedding vector related to the given ids
   */
  operator fun get(tId: Int, pId: Int, dId: Int): DenseNDArray = concatVectorsV(
    this.transitionEmbeddings.get(tId).array.values,
    this.posTagEmbeddings.get(pId).array.values,
    this.deprelEmbeddings.get(dId).array.values
  )

  /**
   * @return the 'nullEmbedding'
   */
  fun getNullVector(): DenseNDArray = concatVectorsV(
    this.transitionEmbeddings.get(null).array.values,
    this.posTagEmbeddings.get(null).array.values,
    this.deprelEmbeddings.get(null).array.values
  )
}
