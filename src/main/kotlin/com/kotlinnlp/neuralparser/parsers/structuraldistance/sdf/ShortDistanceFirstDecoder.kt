/* Copyright 2019-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistance.sdf

import com.kotlinnlp.neuralparser.parsers.structuraldistance.StructuralDistanceParserModel
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.removeAtIndexOfFirst
import com.kotlinnlp.utils.removeFrom
import com.kotlinnlp.utils.toFixed

/**
 * Decode the dependencies using the Short-Distance First (SDF) strategy (Grella, 2019).
 */
class ShortDistanceFirstDecoder(private val model: StructuralDistanceParserModel) {

  /**
   * The PendingElement represent a subtree with the left and the right spines.
   *
   * @param id the id
   * @param vector the vector representation
   * @param position the position in the list of elements
   */
  inner class PendingElement(val id: Int, val vector: DenseNDArray, val position: Int, val word: String) {

    /**
     * The estimated depth of this element in the vector space of the sentence.
     */
    val depth: Double by lazy {
      this@ShortDistanceFirstDecoder.depthProcessor.forward(this.vector).expectScalar()
    }

    /**
     * The estimated depth of this element in the vector space of the sentence.
     */
    val height: Double by lazy {
      this@ShortDistanceFirstDecoder.heightProcessor.forward(this.vector).expectScalar()
    }

    /**
     * Cache of the estimated distances of this element with another element in the sentence vector space.
     */
    private val distanceCache = mutableMapOf<PendingElement, Double>()

    /**
     * Predict the distance this element with the [other] in the sentence vector space.
     *
     * @param other an element
     *
     * @return the distance
     */
    fun distance(other: PendingElement): Double = this.distanceCache.getOrPut(other) {
      this@ShortDistanceFirstDecoder.distanceProcessor.forward(listOf(this.vector, other.vector)).expectScalar()
    }

    /**
     * Whether this element is higher in the predicted hierarchy (i.e. closer to the root node) than
     * the [other] element, or not.
     *
     * @param other an element
     *
     * @return whether this element is higher than the [other], or not
     */
    fun higherThan(other: PendingElement): Boolean = this.depth < other.depth //|| this.height > other.height

    /**
     * The left spine (including itself for convenience)
     */
    val leftSpine: MutableList<PendingElement> = mutableListOf(this)

    /**
     * The right spine (including itself for convenience)
     */
    val rightSpine: MutableList<PendingElement> = mutableListOf(this)

    override fun toString(): String {

      fun str(element: PendingElement) = "${element.id}:${element.word}:d${element.depth.toFixed(2)},h${element.height.toFixed(2)}"

      val ls = this.leftSpine.drop(1).reversed().map(::str)
      val rs = this.rightSpine.drop(1).map(::str)

      return "$ls ${str(this)} $rs"
    }

    override fun equals(other: Any?): Boolean = other is PendingElement && this.id == other.id

    override fun hashCode(): Int = this.id
  }

  /**
   *
   */
  class PendingList(private val list: MutableList<PendingElement>): MutableList<PendingElement> by list {

    companion object {
      operator fun invoke(list: List<PendingElement>) = PendingList(list.toMutableList())
    }

    /**
     * Generate the possible actions.
     *
     * @return the list of possible actions
     */
    fun getPossibleActions(): List<Action> {

      return this.zipWithNext().foldIndexed(mutableListOf()) { i, acc, (first, second) ->

        first.rightSpine.forEachIndexed { k, element ->
          if (element.higherThan(second)) {
            acc.add(Action.ArcRight(pairIndex = i, k = k, distance = element.distance(second)))
          }
        }

        second.leftSpine.forEachIndexed { k, element ->
          if (element.higherThan(first)) {
            acc.add(Action.ArcLeft(pairIndex = i, k = k, distance = element.distance(first)))
          }
        }

        acc
      }
    }

    /**
     * @param action the action to perform
     *
     * @return a triple of governor-id, dependent-id, distance
     */
    fun perform(action: Action): Triple<Int, Int, Double> = when(action) {
      is Action.ArcLeft -> this.perform(action)
      is Action.ArcRight -> this.perform(action)
    }

    /**
     * @param action the action to perform
     *
     * @return a triple of governor-id, dependent-id, distance
     */
    private fun perform(action: Action.ArcLeft): Triple<Int, Int, Double> {

      val pairs = this.zipWithNext()

      val dep = pairs[action.pairIndex].first
      val gov = pairs[action.pairIndex].second

      gov.leftSpine.removeFrom(action.k + 1).addAll(dep.leftSpine)

      this.remove(dep.id)

      return Triple(gov.leftSpine[action.k].id, dep.id, action.distance)
    }

    /**
     * @param action the action to perform
     *
     * @return a triple of governor-id, dependent-id, distance
     */
    private fun perform(action: Action.ArcRight): Triple<Int, Int, Double> {

      val pairs = this.zipWithNext()

      val dep = pairs[action.pairIndex].second
      val gov = pairs[action.pairIndex].first

      gov.rightSpine.removeFrom(action.k + 1).addAll(dep.rightSpine)

      this.remove(dep.id)

      return Triple(gov.rightSpine[action.k].id, dep.id, action.distance)
    }

    /**
     * @param id
     */
    private fun remove(id: Int) { this.removeAtIndexOfFirst { it.id == id } }
  }

  /**
   * The processor to predict the 'tree-depth'.
   */
  private val depthProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = this.model.depthModel,
    useDropout = false,
    propagateToInput = false)

  /**
   * The processor to predict the 'tree-depth'.
   */
  private val heightProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = this.model.heightModel,
    useDropout = false,
    propagateToInput = false)

  /**
   * The processor to predict the 'tree-distance'.
   */
  private val distanceProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = this.model.distanceModel,
    useDropout = false,
    propagateToInput = false)

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param ids the ids of the elements
   * @param vectors the encoded representation of the elements (aligned with the [ids])
   *
   * @return the list of arcs of the form <governor, dependent, score>
   */
  fun decode(ids: List<Int>, vectors: List<DenseNDArray>, words: List<String>): List<Triple<Int, Int, Double>> {

    val arcs = mutableListOf<Triple<Int, Int, Double>>()

    val pendingList = PendingList(ids.zip(vectors).mapIndexed { position, (id, v) ->
      PendingElement(id = id, vector = v, position = position, word = words[position])
    })

    //pendingList.forEach { print("($it) ") }; println() // TODO: remove it

    while (pendingList.size > 1) {

      val possibleActions = pendingList.getPossibleActions()
      val bestAction = this.selectBestAction(possibleActions)
      val newArc = pendingList.perform(bestAction)

      arcs.add(newArc)

      //pendingList.forEach { print("($it) ") }; println() // TODO: remove it
    }

    return arcs
  }

  /**
   * Select the action with the lowest distance.
   *
   * @return the best action
   */
  private fun selectBestAction(actions: List<Action>) = actions.sortedBy { it.distance }.first()
}