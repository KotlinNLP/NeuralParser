package com.kotlinnlp.morphodisambiguator.language

import java.io.Serializable

/**
 * The list of names of all the possible properties.
 */
data class PropertyNames(
    val propertyNames: List<String> = listOf("gender", "number", "person", "mood", "tense")
): Serializable {
  companion object {


    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

  }


}
