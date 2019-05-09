/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.morphodisambiguator.utils

import com.kotlinnlp.morphodisambiguator.language.MorphoDictionary
import com.kotlinnlp.utils.stats.MetricCounter

data class MetricsCounter(
    val corpusMorphologies: MorphoDictionary,
    val correctMorphologyProperties: HashMap<String, Int> = HashMap(),
    val counterMorphologyProperties: HashMap<String, MetricCounter> = HashMap(),
    var totalSentences: Int = 0,
    var totalTokens: Int = 0) {

  init {
    corpusMorphologies.morphologyDictionaryMap.keys.map {
      correctMorphologyProperties.put(it, 0)
      counterMorphologyProperties.put(it, MetricCounter())
    }

  }

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String  {
    var str = ""
    this.counterMorphologyProperties.forEach { k, v -> str = str + k + " " + v.toString() + "\n"}
    return str
  }

}