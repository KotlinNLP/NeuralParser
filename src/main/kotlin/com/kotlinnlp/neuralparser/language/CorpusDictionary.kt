/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.google.common.collect.HashMultimap
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The CorpusDictionary.
 */
class CorpusDictionary : Serializable {

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
    operator fun invoke(sentences: List<CoNLLSentence>): CorpusDictionary {

      val dictionary = CorpusDictionary()

      sentences.forEach { it.tokens.forEach { token -> dictionary.addInfo(token) } }

      return dictionary
    }
  }

  /**
   * The words.
   */
  val words = DictionarySet<String>()

  /**
   * The map of forms to their possible POS tags.
   */
  val formsToPosTags: HashMultimap<String, List<POSTag>> = HashMultimap.create()

  /**
   * The dictionary set of all the possible grammatical configurations.
   */
  val grammaticalConfigurations = DictionarySet<GrammaticalConfiguration>()

  /**
   * Add the info of a given [token] into this dictionary.
   *
   * @param token the token of a sentence
   */
  private fun addInfo(token: CoNLLToken) {

    this.words.add(token.normalizedForm)

    this.formsToPosTags.put(token.normalizedForm, token.posList)

    this.grammaticalConfigurations.add(GrammaticalConfiguration(*Array(
      size = maxOf(token.posList.size, token.syntacticDependencies.size),
      init = { i -> GrammaticalConfiguration.Component(
        pos = token.posList.getOrElse(i) { token.posList.single() },
        syntacticDependency = token.syntacticDependencies.getOrElse(i) { token.syntacticDependencies.single() })
      })))
  }
}
