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
class PointerDistanceDecoder(model: DistanceParserModel<PointerDistanceDecoder>) : DependencyDecoder(model) {

  /**
   * An encoded token, providing a method to calculate the rounded distance with another.
   *
   * @param id the id
   * @param vector the vector representation
   */
  private inner class RoundedDistToken(id: Int, vector: DenseNDArray) : Token(id = id, vector = vector) {

    /**
     * @param other another token
     *
     * @return the structural distance between this token and the [other], rounded to the nearest non-zero positive
     *         integer
     */
    fun roundedDistance(other: Token): Int = this@PointerDistanceDecoder.getRoundedDistance(this, other)
  }

  /**
   * Cache of the estimated rounded distances of a token with another.
   */
  private val distanceCache: MutableMap<Token, MutableMap<Token, Int>> = mutableMapOf()

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param indexedVectors the encoded representation of the tokens with the related ids
   *
   * @return the list of arcs as triples of the form <governor, dependent, score>
   */
  override fun decode(indexedVectors: List<IndexedValue<DenseNDArray>>): List<Triple<Int, Int, Double>> {

    val tokens: List<RoundedDistToken> = indexedVectors.map { RoundedDistToken(id = it.index, vector = it.value) }

    val pendingList: MutableList<RoundedDistToken> = tokens.toMutableList()
    val root: RoundedDistToken = tokens.minBy { it.depth }!!

    val arcs: MutableList<Triple<Int, Int, Double>> = mutableListOf()
    var leaves: List<RoundedDistToken> = listOf(root)

    pendingList.remove(root)

    while (pendingList.size > 1) {

      if (leaves.isNotEmpty()) {

        val newLeaves: MutableList<RoundedDistToken> = mutableListOf()

        leaves.forEach { leaf ->

          pendingList
            .asSequence()
            .filter { it.roundedDistance(leaf) == 1 }
            .forEach { dependent ->

              val score: Double = 1.0 - dependent.distance(leaf)

              arcs.add(Triple(leaf.id, dependent.id, score))
              newLeaves.add(dependent)
              pendingList.remove(dependent)
            }
        }

        leaves = newLeaves

      } else {

        // Note: it is necessary to make a copy of the pendingList in order to iterate it while removing its elements
        pendingList.toList().forEach { remainingToken ->

          val governor: Token = tokens.minBy { it.distance(remainingToken) }!!
          val score: Double = 1.0 - remainingToken.distance(governor)

          arcs.add(Triple(governor.id, remainingToken.id, score))
          pendingList.remove(remainingToken)
        }
      }
    }

    return arcs
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
