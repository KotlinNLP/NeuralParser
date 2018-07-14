/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.statistics

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
    - Labeled   attachment score:          %5.2f %% (%d / %d)
    - Unlabeled attachment score:          %5.2f %% (%d / %d)
    - Deprel  accuracy score:              %5.2f %% (%d / %d)
    - POS tag accuracy score:              %5.2f %% (%d / %d)
    - Sentence labeled   attachment score: %5.2f %% (%d / %d)
    - Sentence unlabeled attachment score: %5.2f %% (%d / %d)
    """
    .removePrefix("\n")
    .trimIndent()
    .format(
      this.las.perc100, this.las.count, this.las.total,
      this.uas.perc100, this.uas.count, this.uas.total,
      this.ds.perc100, this.ds.count, this.ds.total,
      this.ps.perc100, this.ps.count, this.ps.total,
      this.slas.perc100, this.slas.count, this.slas.total,
      this.suas.perc100, this.suas.count, this.suas.total
    )
}
