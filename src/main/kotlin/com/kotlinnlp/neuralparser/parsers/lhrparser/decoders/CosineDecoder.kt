/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.decoders

import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.HeadsDecoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.LatentSyntacticStructure
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.ArcScores
import com.kotlinnlp.simplednn.simplemath.cosineSimilarity

/**
 * The HeadsDecoder.
 */
class CosineDecoder : HeadsDecoder {

  /**
   * The private map of scored arcs.
   * Scores are mapped by dependents to governors ids (the root is intended to have id = -1).
   */
  private val similarityMatrix = mutableMapOf<Int, MutableMap<Int, Double>>()

  /**
   * The latent syntactic structure that contains the context-vectors and latent-heads used in the calculation of
   * the similarity.
   *
   * The root vector must be normalized each time because it is being trained.
   */
  private lateinit var lssNorm: LatentSyntacticStructure

  /**
   * Calculate the similarity scores among the context-vectors, the latent-heads and the root-vector.
   *
   * @param lss the latent syntactic structure to decode
   *
   * @return the computed scores
   */
  override fun decode(lss: LatentSyntacticStructure): ArcScores {

    this.lssNorm = LatentSyntacticStructure(
      sentence = lss.sentence,
      contextVectors = lss.contextVectors.map { it.normalize2() },
      latentHeads = lss.latentHeads.map { it.normalize2() },
      virtualRoot = lss.virtualRoot.normalize2())

    lss.sentence.tokens.forEach {

      this.similarityMatrix[it.id] = mutableMapOf()

      this.setHeadsScores(it)
      this.setRootScore(it)
    }

    return ArcScores(scores = this.similarityMatrix)
  }

  /**
   * Set the heads scores of the given [dependent] in the [similarityMatrix] map.
   *
   * @param dependent the the dependent token
   */
  private fun setHeadsScores(dependent: ParsingToken) {

    val scores: MutableMap<Int, Double> = this.similarityMatrix.getValue(dependent.id)

    this.lssNorm.sentence.tokens
      .filter { it.id != dependent.id }
      .associateTo(scores) {
        it.id to cosineSimilarity(
          a = this.lssNorm.getContextVectorById(it.id),
          b = this.lssNorm.getLatentHeadById(dependent.id))
      }
  }

  /**
   * Set the root score of the given [dependent] in the [similarityMatrix] map.
   *
   * @param dependent the the dependent token
   */
  private fun setRootScore(dependent: ParsingToken) {

    this.similarityMatrix.getValue(dependent.id)[ArcScores.rootId] = 0.0 // default root score

    if (!dependent.isPunctuation) { // the root shouldn't be a punctuation token

      this.similarityMatrix.getValue(dependent.id)[ArcScores.rootId] = cosineSimilarity(
        a = this.lssNorm.getLatentHeadById(dependent.id),
        b = this.lssNorm.virtualRoot)
    }
  }
}
