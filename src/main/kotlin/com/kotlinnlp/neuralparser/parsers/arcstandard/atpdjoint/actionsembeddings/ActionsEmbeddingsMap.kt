/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.arcstandard.atpdjoint.actionsembeddings

import com.kotlinnlp.simplednn.deeplearning.embeddings.Embedding
import com.kotlinnlp.simplednn.deeplearning.embeddings.EmbeddingsMap
import com.kotlinnlp.simplednn.utils.MultiMap
import java.io.Serializable

/**
 * The container of embeddings that encode actions, mapped to their TPD ids.
 *
 * @property size the size of each embedding vector
 * @property transitionsSize the number of transitions
 * @property posTagsSize the number of POS tags
 * @property deprelsSize the number of deprels
 */
class ActionsEmbeddingsMap(
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
   * The embeddings map used only to extract the 'nullEmbedding', needed to encode the 'null' action.
   */
  val nullEmbeddingMap = EmbeddingsMap<Int>(size = this.size)

  /**
   * A nested multimap that maps transition, POS tag and deprel ids to embeddings vectors.
   */
  private val tpdMultimap: MultiMap<EmbeddingsMap<Int>> = MultiMap(mapOf(
    *(0 until this.transitionsSize).map { tId ->
      // transition level
      Pair(
        tId,
        mapOf(*(0 until this.posTagsSize).map { pId ->
          // POS tag level
          Pair(
            pId,
            EmbeddingsMap<Int>(size = this.size) // deprel level
          )
        }.toTypedArray())
      )
    }.toTypedArray()
  ))

  /**
   * Check sizes and initialize the deprels embeddings maps.
   */
  init {
    require(this.size > 0) { "The embeddings vectors size must be > 0" }
    require(this.transitionsSize > 0) { "The transitions size must be > 0" }
    require(this.posTagsSize > 0) { "The POS tags size must be > 0" }
    require(this.deprelsSize > 0) { "The deprels size must be > 0" }

    this.tpdMultimap.forEach { _, _, deprelEmbeddingsMap ->
      (0 until this.deprelsSize).forEach { dId -> deprelEmbeddingsMap.set(dId)  }
    }
  }

  /**
   * @param tId the transition id
   * @param pId the POS tag id
   *
   * @throws KotlinNullPointerException if [tId] is not within the range [0, [transitionsSize])
   * @throws KotlinNullPointerException if [pId] is not within the range [0, [posTagsSize])
   *
   * @return the embeddings map related to the given ids
   */
  operator fun get(tId: Int, pId: Int): EmbeddingsMap<Int> = this.tpdMultimap[tId, pId]!!

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
  operator fun get(tId: Int, pId: Int, dId: Int): Embedding = this[tId, pId].get(dId)

  /**
   * @return the 'nullEmbedding'
   */
  fun getNullEmbedding() = this.nullEmbeddingMap.nullEmbedding
}
