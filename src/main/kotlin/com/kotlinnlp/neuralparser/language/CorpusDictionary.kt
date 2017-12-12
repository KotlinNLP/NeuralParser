/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.google.common.collect.HashMultimap
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import com.kotlinnlp.simplednn.utils.DictionarySet
import java.io.Serializable

/**
 * The CorpusDictionary.
 */
class CorpusDictionary : Serializable {

  companion object Factory {

    /**
     * Use the given [sentences] to populate the dictionary sets (words, posTags, deprelTags).
     *
     * @param sentences a list of [Sentence]
     */
    operator fun invoke(sentences: List<Sentence>): CorpusDictionary {

      val dictionary = CorpusDictionary()

      sentences.forEach { sentence ->

        sentence.tokens.forEach { token -> dictionary.addInfo(token) }

        if (sentence.dependencyTree != null) {
          dictionary.addDependencies(sentence.dependencyTree)
        }
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
   * Add the dependencies of a given [dependencyTree] into this dictionary.
   *
   * @param dependencyTree the dependency tree of a sentence
   */
  private fun addDependencies(dependencyTree: DependencyTree) {

    dependencyTree.deprels.zip(dependencyTree.posTags).forEach { (deprel, posTag) ->

      if (deprel != null) {
        this.deprelTags.add(deprel)

        if (posTag != null) {
          this.deprelPosTagCombinations.put(deprel, posTag)
        }
      }
    }
  }

  /**
   * Add info (form and POS) of a given [token] into this dictionary.
   *
   * @param token the token of a sentence
   */
  private fun addInfo(token: Token) {

    val posTag = POSTag(token.pos!!)

    this.words.add(token.normalizedWord)
    this.posTags.add(posTag)
    this.formsToPosTags.put(token.normalizedWord, posTag)
  }
}
