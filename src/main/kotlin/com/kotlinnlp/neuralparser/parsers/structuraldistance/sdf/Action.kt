/* Copyright 2019-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistance.sdf

/**
 * The Action.
 *
 * @property pairIndex the index of the pair from which to extract the governor and the dependent
 * @property k the index of the left or right spine from which to add new elements
 * @property distance the predicted distance between the governor and the dependent
 */
sealed class Action(val pairIndex: Int, val k: Int, val distance: Double) {

  /**
   * The ArcLeft action.
   *
   * @param pairIndex the index of the pair from which to extract the governor and the dependent
   * @param k the index of the left spine from which to add new elements
   * @param distance the predicted distance between the governor and the dependent
   */
  class ArcLeft(pairIndex: Int, k: Int, distance: Double) : Action(pairIndex, k, distance)

  /**
   * The ArcRight action.
   *
   * @param pairIndex the index of the pair from which to extract the governor and the dependent
   * @param k the index of the right spine from which to add new elements
   * @param distance the predicted distance between the governor and the dependent
   */
  class ArcRight(pairIndex: Int, k: Int, distance: Double) : Action(pairIndex, k, distance)
}