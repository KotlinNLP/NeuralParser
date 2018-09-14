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
   * The set of all the possible POS tags.
   */
  val posTags = DictionarySet<POSTag>()

  /**
   * The set of all the possible deprels.
   */
  val deprels = DictionarySet<Deprel>()

  /**
   * The map of forms to their possible POS tags.
   */
  val formsToPosTags: HashMultimap<String, POSTag> = HashMultimap.create()

  /**
   * The dictionary set of all the possible dependency relations.
   */
  val dependencyRelations = DictionarySet<DependencyRelation>()

  /**
   * Add the info of a given [token] into this dictionary.
   *
   * @param token the token of a sentence
   */
  private fun addInfo(token: CoNLLToken) {

    val posTag = POSTag(token.pos)
    val deprel: Deprel = token.buildDeprel()

    this.words.add(token.normalizedForm)
    this.posTags.add(posTag)
    this.deprels.add(deprel)

    this.formsToPosTags.put(token.normalizedForm, posTag)
    this.dependencyRelations.add(DependencyRelation(posTag = posTag, deprel = deprel))
  }

  /**
   * @return the deprel of this token
   */
  private fun CoNLLToken.buildDeprel(): Deprel {

    val head: Int = checkNotNull(this.head)

    return Deprel(
      label = this.deprel,
      direction = when {
        head == 0 -> Deprel.Direction.ROOT
        head > this.id -> Deprel.Direction.LEFT // the id follows the incremental order of the CoNLL sentence tokens
        else -> Deprel.Direction.RIGHT
      })
  }
}
