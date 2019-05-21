/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.distance

import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Decode the dependencies among tokens.
 *
 * @param model the model of a distance parser
 */
abstract class DependencyDecoder(model: DistanceParserModel) {

  /**
   * The processor to predict the 'tree-distance'.
   */
  private val distanceProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = model.distanceModel,
    useDropout = false,
    propagateToInput = false)

  /**
   * The processor to predict the 'tree-depth'.
   */
  private val depthProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = model.depthModel,
    useDropout = false,
    propagateToInput = false)

  /**
     * Cache of the estimated distances of a token with another.
   */
  private val distanceCache: MutableMap<Token, MutableMap<Token, Double>> = mutableMapOf()

  /**
   * The Element.
   *
   * @param id the id
   * @param vector the vector representation
   */
  protected inner class Token(val id: Int, val vector: DenseNDArray) {

    /**
     * The estimated depth of this element in the sentence vector space.
     */
    val depth: Double by lazy { this@DependencyDecoder.depthProcessor.forward(this.vector).expectScalar() }

    /**
     * @param other another token
     *
     * @return the structural distance between this token and the [other]
     */
    fun distance(other: Token): Double = this@DependencyDecoder.getDistance(this, other)

    override fun equals(other: Any?): Boolean = other is Token && this.id == other.id

    override fun hashCode(): Int = this.id
  }

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param indexedVectors the encoded representation of the tokens with the related ids
   *
   * @return the list of arcs as triples of the form <governor, dependent, score>
   */
  abstract fun decode(indexedVectors: List<IndexedValue<DenseNDArray>>): List<Triple<Int, Int, Double>>
  /**
   * @param a a token
   * @param b a token
   *
   * @return the (non directional) distance between [a] and [b]
   */
  protected fun getDistance(a: Token, b: Token): Double {

    val firstToken: Token = sequenceOf(a, b).minBy { it.id }!!
    val secondToken: Token = sequenceOf(a, b).maxBy { it.id }!!

    return this.distanceCache.getOrPut(firstToken) { mutableMapOf() }.getOrPut(secondToken) {
       this.distanceProcessor.forward(listOf(firstToken.vector, secondToken.vector)).expectScalar()
    }
  }
}
