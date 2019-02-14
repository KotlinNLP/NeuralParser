package com.kotlinnlp.morphodisambiguator.helpers.dataset

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Klaxon
import com.kotlinnlp.linguisticdescription.morphology.properties.Degree
import java.io.File
import java.lang.StringBuilder

/**
 * A dataset that can be read from a JSON file and it is used to create a [PreprocessedDataset].
 *
 * @property examples the list of examples
 */
class Dataset (
    val examples: List<Sentence>
) {

  /**
   * Raised when the configuration of the dataset is not valid.
   *
   * @param message the exception message
   */
  class InvalidConfiguration(message: String) : RuntimeException(message)

  /**
   * Raised when an example is not valid.
   *
   * @param index the example index
   * @param message the exception message
   */
  class InvalidExample(index: Int, message: String) : RuntimeException("[Example #$index] $message")

  /**
   * An example of the dataset.
   *
   * @property tokens the list of tokens that compose the sentence of this example
   */
  data class Sentence(val tokens: List<Token>?) {

    /**
     * A token of the example.
     *
     * @property form the token form
     */
    data class Token(val form: String, val lemma: List<Lemma>, val properties: List<Map<String, String?>>)

    data class Lemma(val lemma: String)

  }

  companion object {

    /**
     * Load a [Dataset] from a JSON file with the following template:
     *
     * @param filePath the file path of the JSON file
     *
     * @return the dataset read from the given file
     */

    fun fromFile(filename: String):Dataset {

      val sentences = mutableListOf<Sentence>()

      File(filename).reader().forEachLine { line ->

        val parsedExample = Klaxon().parseArray<Sentence.Token>(line)

        sentences.add(Sentence(tokens=parsedExample))
      }

      return Dataset(examples=sentences)
    }
  }


}
