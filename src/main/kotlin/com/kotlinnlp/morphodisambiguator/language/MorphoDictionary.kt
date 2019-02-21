package com.kotlinnlp.morphodisambiguator.language

import com.google.common.collect.HashMultimap
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

      dictionary.morphologyPropertiesMap.keys().map{
        dictionary.morphologyDictionaryMap.put(it, DictionarySet(dictionary.morphologyPropertiesMap.get(it).toList()))
      }

      return dictionary
    }
  }

  /**
   * The words.
   */
  private val words = DictionarySet<String>()

  /**
   * The map of morphology properties to their possible values.
   */
  private val morphologyPropertiesMap: HashMultimap<String, String> = HashMultimap.create()

  /**
   * The map of morphology properties to an indexed dictionary containing their values
   */
  val morphologyDictionaryMap: HashMap<String, DictionarySet<String>> = HashMap()

  /**
   * The list of names of all the possible properties.
   */
  val propertyNames = PropertyNames().propertyNames

  /**
   * Add the info of a given [token] into this dictionary.
   *
   * @param token the token of a sentence
   */
  private fun addInfo(token: MorphoToken) {

    this.words.add(token.form)
    token.morphoProperties.zip(propertyNames).forEach { (property, name) ->
      if (property != PropertyNames().unknownAnnotation) this.morphologyPropertiesMap.put(name, property)
    }

  }

}
