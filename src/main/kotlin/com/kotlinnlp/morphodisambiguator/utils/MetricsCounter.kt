package com.kotlinnlp.morphodisambiguator.utils

import com.kotlinnlp.morphodisambiguator.language.MorphoDictionary
import com.kotlinnlp.utils.stats.StatMetric

data class MetricsCounter(
    val corpusMorphologies: MorphoDictionary,
    val correctMorphologyProperties: HashMap<String, Int> = HashMap(),
    var totalSentences: Int = 0,
    var totalTokens: Int = 0) {

  init {
    corpusMorphologies.morphologyDictionaryMap.keys.map {
      correctMorphologyProperties.put(it, 0)
    }
  }

  /**
   * @return the base statistics
   */
  fun toStatistics(): HashMap<String, StatMetric> {
    val morphologiesAccuracy : HashMap<String, StatMetric> = HashMap()

    correctMorphologyProperties.keys.map {
      morphologiesAccuracy.put(it, StatMetric(count = correctMorphologyProperties.getValue(it), total = totalTokens))
    }

    return morphologiesAccuracy

  }

  /**
   * @return this statistics formatted into a string
   */
  override fun toString(): String  {
    var str = ""
    this.toStatistics().forEach { k, v -> str = str + k + " " + v.toString() + "\n"}
    return str
  }

}