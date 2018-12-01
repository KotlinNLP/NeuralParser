/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.statistics

/**
 * A single statistic metric.
 *
 * @property count the count of correct elements
 * @property total the total amount of elements
 */
data class StatMetric(val count: Int, val total: Int) {

  /**
   * The percentage of [count] against the [total].
   */
  val perc: Double = count.toDouble() / total

  /**
   * @return the string representation of this metric
   */
  override fun toString(): String = "%5.2f %% (%d / %d)".format(100 * this.perc, this.count, this.total)
}
