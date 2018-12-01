/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.statistics

import com.kotlinnlp.utils.StatMetric

/**
 * A counter of statistic metrics.
 *
 * @property labeledAttachments the counter of labeled attachments
 * @property unlabeledAttachments the counter of unlabeled attachments
 * @property correctPOSTags the counter of correct POS tags
 * @property correctDeprels the counter of correct deprels
 * @property correctLabeledSentences the counter of correct labeled sentences
 * @property correctUnlabeledSentences the counter of correct unlabeled sentences
 * @property totalSentences the total amount of sentences
 * @property totalTokens the total amount of tokens
 */
data class MetricsCounter(
  var labeledAttachments: Int = 0,
  var unlabeledAttachments: Int = 0,
  var correctPOSTags: Int = 0,
  var correctDeprels: Int = 0,
  var correctLabeledSentences: Int = 0,
  var correctUnlabeledSentences: Int = 0,
  var totalSentences: Int = 0,
  var totalTokens: Int = 0) {

  /**
   * @return the base statistics
   */
  fun toStatistics() = with(this) {
    BaseStatistics(
      las = StatMetric(count = labeledAttachments, total = totalTokens),
      uas = StatMetric(count = unlabeledAttachments, total = totalTokens),
      ps = StatMetric(count = correctPOSTags, total = totalTokens),
      ds = StatMetric(count = correctDeprels, total = totalTokens),
      slas = StatMetric(count = correctLabeledSentences, total = totalSentences),
      suas = StatMetric(count = correctUnlabeledSentences, total = totalSentences)
    )
  }
}
