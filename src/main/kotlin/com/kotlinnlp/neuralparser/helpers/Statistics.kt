/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

/**
 * Parsing statistics, including ones calculated without considering the punctuation.
 *
 * @property las labeled attachment score
 * @property uas unlabeled attachment score
 * @property ps POS tag accuracy score
 * @property ds deprel accuracy score
 * @property slas sentence labeled attachment score
 * @property suas sentence unlabeled attachment score
 * @property noPunctuation statistics without considering punctuation tokens
 */
class Statistics(
  las: StatMetric,
  uas: StatMetric,
  ps: StatMetric,
  ds: StatMetric,
  slas: StatMetric,
  suas: StatMetric,
  val noPunctuation: BaseStatistics
) : BaseStatistics(las = las, uas = uas, ps = ps, ds = ds, slas = slas, suas = suas) {

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String = """
    Evaluation stats:
    %s

    Evaluation stats without considering punctuation:
    %s
    """
    .removePrefix("\n")
    .trimIndent()
    .format(
      super.toString(),
      this.noPunctuation.toString()
    )
}
