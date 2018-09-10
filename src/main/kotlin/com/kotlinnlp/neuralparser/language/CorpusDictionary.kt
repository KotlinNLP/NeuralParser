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
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.utils.DictionarySet
import java.io.Serializable

/**
 * The CorpusDictionary.
 */
class CorpusDictionary : Serializable {

  companion object Factory {

    /**
     * Use the given [sentences] to populate the dictionary sets (words, posTags, deprelTags).
     *
     * @param sentences a list of sentences
     *
     * @return a new CorpusDictionary
     */
    operator fun invoke(sentences: List<CoNLLSentence>): CorpusDictionary {

      val dictionary = CorpusDictionary()

      sentences.forEach { sentence ->
        sentence.tokens.forEach { token -> dictionary.addInfo(token) }
      }

      return dictionary
    }
  }

  /**
   * The words.
   */
  val words = DictionarySet<String>()

  /**
   * The pos tags.
   */
  val posTags = DictionarySet<POSTag>()

  /**
   * The deprel tags.
   */
  val deprelTags = DictionarySet<Deprel>()

  /**
   * The map of forms to pos tags.
   */
  val formsToPosTags: HashMultimap<String, POSTag> = HashMultimap.create()

  /**
   * The map of forms to pos tags.
   */
  val deprelPosTagCombinations: HashMultimap<Deprel, POSTag> = HashMultimap.create()

  /**
   * Add info (form and POS) of a given [token] into this dictionary.
   *
   * @param token the token of a sentence
   */
  private fun addInfo(token: CoNLLToken) {

    val posTag = POSTag(token.pos)
    val deprel: Deprel = this.getDeprel(token)

    this.words.add(token.normalizedForm)
    this.posTags.add(posTag)
    this.formsToPosTags.put(token.normalizedForm, posTag)

    if (token.head != null) {
      this.deprelTags.add(deprel)
      this.deprelPosTagCombinations.put(deprel, posTag)
    }
  }

  /**
   * @param token the token of a sentence
   *
   * @return the deprel of the given [token]
   */
  private fun getDeprel(token: CoNLLToken): Deprel {

    val head: Int = checkNotNull(token.head)

    return Deprel(
      label = token.deprel,
      direction = when {
        head == 0 -> Deprel.Direction.ROOT
        head > token.id -> Deprel.Direction.LEFT // the id follows the incremental order of the CoNLL sentence tokens
        else -> Deprel.Direction.RIGHT
      })
  }
}
