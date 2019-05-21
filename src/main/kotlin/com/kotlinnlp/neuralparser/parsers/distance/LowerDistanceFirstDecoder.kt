/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.distance

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Decode the dependencies using the Lower-Distance First (LDF) parsing strategy (Grella, 2019).
 *
 * @param model the model of a distance parser
 */
class LowerDistanceFirstDecoder(model: DistanceParserModel<LowerDistanceFirstDecoder>) : DependencyDecoder(model) {

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param indexedVectors the encoded representation of the tokens with the related ids
   *
   * @return the list of arcs as triples of the form <governor, dependent, score>
   */
  override fun decode(indexedVectors: List<IndexedValue<DenseNDArray>>): List<Triple<Int, Int, Double>> {

    val arcs = mutableListOf<Triple<Int, Int, Double>>()
    val tokens: List<Token> = indexedVectors.map { Token(id = it.index, vector = it.value) }
    val pendingList: MutableList<Token> = tokens.toMutableList()

    while (pendingList.size > 1) {

      val pairs = pendingList.zipWithNext()

      val (first, second, distance) = this.selectBest(pairs)

      val dep: Token
      val gov: Token

      if (first.depth > second.depth) {
        dep = first
        gov = second
      } else {
        dep = second
        gov = first
      }

      arcs.add(Triple(gov.id, dep.id, distance))

      pendingList.removeAtIndexOfFirst { it.id == dep.id }
    }

    return arcs
  }

  /**
   * Select the pair with the lowest distance.
   *
   * @return a triple <first, second, distance>
   */
  private fun selectBest(pairs: List<Pair<Token, Token>>): Triple<Token, Token, Double> = pairs
    .sortedBy { (a, b) -> a.distance(b) }
    .first()
    .let { (a, b) -> Triple(a, b, a.distance(b)) }

  /**
   * Removes an element at the position of the first element that matches the given [predicate].
   *
   * @return the element that has been removed.
   */
  private fun <T> MutableList<T>.removeAtIndexOfFirst(predicate: (T) -> Boolean)
    = this.removeAt(this.indexOfFirst(predicate))
}
