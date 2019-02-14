package com.kotlinnlp.morphodisambiguator.language

import com.google.common.collect.ArrayListMultimap
import com.google.common.collect.ListMultimap
import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The Morphological properties Dictionary.
 */
class MorphoDictionary: Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Create a new corpus populated with the information contained in the given [sentences] (words, POS tags and
     * deprels).
     *
     * @param sentences a list of sentences
     *
     * @return a new corpus dictionary
     */
    operator fun invoke(sentences: List<PreprocessedDataset.MorphoSentence>): MorphoDictionary {

      val dictionary = MorphoDictionary()

      sentences.forEach { it.tokens.forEach { token -> dictionary.addInfo(token) } }

      return dictionary
    }
  }

  /**
   * The words.
   */
  val words = DictionarySet<String>()

  /**
   * The map of morphologies to their possible values.
   */
  val morphologyPropertiesMap: ListMultimap<String, String> = ArrayListMultimap.create()


  /**
   * Add the info of a given [token] into this dictionary.
   *
   * @param token the token of a sentence
   */
  private fun addInfo(token: MorphoToken) {

    this.words.add(token.form)
    token.morphoProperties.forEach{
      it.forEach{property -> this.morphologyPropertiesMap.put(property.toString(), property?.annotation)}
    }

  }
}
