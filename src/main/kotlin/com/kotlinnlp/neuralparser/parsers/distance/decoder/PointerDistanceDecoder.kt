/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.distance.decoder

import com.kotlinnlp.neuralparser.parsers.distance.DistanceParserModel
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import kotlin.math.roundToInt

/**
 * Decode the dependencies using a Pointer Network to detect the Top token and assigning the dependents in depth with
 * a greedy approach that takes advantage of a Distance Layer (Cangialosi, 2019).
 *
 * @param model the model of a distance parser
 */
class PointerDistanceDecoder(model: DistanceParserModel) : DependencyDecoder(model) {

  /**
   * An encoded token, providing a method to calculate the rounded distance with another.
   *
   * @param id the id
   * @param vector the vector representation
   */
  private inner class RoundedDistToken(id: Int, vector: DenseNDArray) : Token(id = id, vector = vector) {

    /**
     * The cache map for the best direct governor of this token.
     * It is null if there is no a direct governor.
     *
     * (A direct governor is a token which predicted rounded distance from this is 1).
     */
    private val bestDirectGovernorCache: MutableMap<Int, RoundedDistToken?> = mutableMapOf()

    /**
     * @param other another token
     *
     * @return the structural distance between this token and the [other], rounded to the nearest non-zero positive
     *         integer
     */
    fun roundedDistance(other: Token): Int = this@PointerDistanceDecoder.getRoundedDistance(this, other)

    /**
     * Get the best direct governor of this token or null if there is no a direct governor.
     *
     * (A direct governor is a token which predicted rounded distance from this is 1).
     *
     * @param tokens the list of all the tokens of the sentence
     *
     * @return the best direct governor of this token or null if there is no governor
     */
    fun getBestDirectGovernor(tokens: List<RoundedDistToken>): RoundedDistToken? =
      this.bestDirectGovernorCache.getOrPut(0) {
        tokens
          .filter { it != this && it.roundedDistance(this) == 1 }
          .maxBy { it.attachmentScore(this) }
      }

    /**
     * @param tokens a list of tokens
     *
     * @return the nearest token within a given list of tokens
     */
    fun getNearest(tokens: List<RoundedDistToken>): RoundedDistToken =
      tokens.filter { it != this }.minBy { it.distance(this) }!!

    /**
     * Get the attachment score between this token and the [other].
     * The rounded distance between the two tokens is intended to be 1.
     *
     * @param other another token
     *
     * @return the attachment score between this token and the [other]
     */
    fun attachmentScore(other: RoundedDistToken): Double =
      1.0 - Math.abs(this.distance(other) - this.roundedDistance(other))
  }

  /**
   * Cache of the estimated rounded distances of a token with another.
   */
  private val distanceCache: MutableMap<Token, MutableMap<Token, Int>> = mutableMapOf()

  /**
   * The tokens not already attached.
   * It is a support variable used during the decoding.
   */
  private lateinit var pendingList: MutableList<RoundedDistToken>

  /**
   * All the attached tokens.
   * It is a support variable used during the decoding.
   */
  private lateinit var attachedTokens: MutableList<RoundedDistToken>

  /**
   * The last attached tokens.
   * It is a support variable used during the decoding.
   */
  private lateinit var lastAttachedTokens: List<RoundedDistToken>

  /**
   * The arcs that has been decoded.
   * It is a support variable used during the decoding.
   */
  private lateinit var arcs: MutableList<Triple<Int, Int, Double>>

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param indexedVectors the encoded representation of the tokens with the related ids
   *
   * @return the list of arcs as triples of the form <governor, dependent, score>
   */
  override fun decode(indexedVectors: List<IndexedValue<DenseNDArray>>): List<Triple<Int, Int, Double>> {

    val tokens: List<RoundedDistToken> = indexedVectors.map { RoundedDistToken(id = it.index, vector = it.value) }

    this.initDecoding(tokens)

    while (this.pendingList.isNotEmpty()) {

      if (this.lastAttachedTokens.isNotEmpty())
        this.attachDirectTokens(tokens)
      else
        this.attachNearestToken()

      this.attachedTokens.addAll(this.lastAttachedTokens)
    }

    return this.arcs
  }

  /**
   * Initialize the support variables for the decoding of a sentence.
   *
   * @param tokens the tokens of the sentence
   */
  private fun initDecoding(tokens: List<RoundedDistToken>) {

    val root: RoundedDistToken = tokens.minBy { it.depth }!!

    this.pendingList = (tokens - root).toMutableList()
    this.attachedTokens = mutableListOf(root)
    this.lastAttachedTokens = listOf(root)
    this.arcs = mutableListOf(Triple(root.id, root.id, 1.0 - Math.abs(root.depth)))
  }

  /**
   * Attach the tokens of the [pendingList] that have a direct governor among the last attached tokens.
   *
   * @param tokens the tokens of the sentence
   */
  private fun attachDirectTokens(tokens: List<RoundedDistToken>) {

    this.lastAttachedTokens = this.lastAttachedTokens.flatMap { attachedToken ->
      this.pendingList
        .filter { it.roundedDistance(attachedToken) == 1 && it.getBestDirectGovernor(tokens) == attachedToken }
        .map { dependent ->
          this.pendingList.remove(dependent)
          this.arcs.add(Triple(attachedToken.id, dependent.id, dependent.attachmentScore(attachedToken)))
          dependent
        }
    }
  }

  /**
   * Attach the token of the [pendingList] that is the nearest to one of the [attachedTokens].
   */
  private fun attachNearestToken() {

    val nearestRemaining: RoundedDistToken =
      this.pendingList.minBy { it.distance(it.getNearest(this.attachedTokens)) }!!

    val governor: Token = nearestRemaining.getNearest(this.attachedTokens)
    val score: Double = 1.0 - nearestRemaining.distance(governor)

    this.pendingList.remove(nearestRemaining)
    this.arcs.add(Triple(governor.id, nearestRemaining.id, score))

    this.lastAttachedTokens = listOf(nearestRemaining)
  }

  /**
   * @param a a token
   * @param b a token
   *
   * @return the (non directional) distance between [a] and [b], rounded to the nearest non-zero positive integer
   */
  private fun getRoundedDistance(a: Token, b: Token): Int {

    val firstToken: Token = sequenceOf(a, b).minBy { it.id }!!
    val secondToken: Token = sequenceOf(a, b).maxBy { it.id }!!

    return this.distanceCache.getOrPut(firstToken) { mutableMapOf() }.getOrPut(secondToken) {
      this.getDistance(firstToken, secondToken).roundToInt().coerceAtLeast(1)
    }
  }
}
