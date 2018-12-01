/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.statistics

import com.kotlinnlp.utils.stats.StatMetric

/**
 * Base parsing statistics.
 *
 * @property las labeled attachment score
 * @property uas unlabeled attachment score
 * @property ps POS tag accuracy score
 * @property ds deprel accuracy score
 * @property slas sentence labeled attachment score
 * @property suas sentence unlabeled attachment score
 */
open class BaseStatistics(
  val las: StatMetric,
  val uas: StatMetric,
  val ps: StatMetric,
  val ds: StatMetric,
  val slas: StatMetric,
  val suas: StatMetric) {

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String = """
    - Labeled   attachment score:          $las
    - Unlabeled attachment score:          $uas
    - Deprel  accuracy score:              $ds
    - POS tag accuracy score:              $ps
    - Sentence labeled   attachment score: $slas
    - Sentence unlabeled attachment score: $suas
    """
    .removePrefix("\n")
    .trimIndent()
}
