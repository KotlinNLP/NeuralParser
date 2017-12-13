/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.beust.klaxon.JsonObject
import com.beust.klaxon.json
import com.kotlinnlp.conllio.MultiWord
import com.kotlinnlp.dependencytree.DependencyTree
import java.io.Serializable

/**
 * The Token.
 *
 * @property id the token id
 * @property word the word form
 * @property pos the part-of-speech tag
 * @property multiWord the multi-word information
 */
class Token(
  val id: Int,
  val word: String,
  var pos: String? = null,
  val multiWord: MultiWord? = null
) : Serializable {

  companion object {

    /**
     * Regex pattern for numeric values.
     */
    val numberRegex = Regex("^\\d+(?:[,.]\\d+)*(?:\\[.,]\\d+)?\$")

    /**
     * Regex pattern for comma and semi-colon.
     */
    val punctuationRegex = Regex("^[,;]+\$")
  }

  /**
   * Whether the word is a numeric values.
   */
  val isNumber: Boolean by lazy { numberRegex.matches(this.word) }

  /**
   * Whether the word is comma or semi-colon.
   */
  val isPunctuation: Boolean by lazy { punctuationRegex.matches(this.word) }

  /**
   * The normalized word-form (lower case and numbers replaced with a common form)
   */
  val normalizedWord: String get() = if (this.isNumber) "__num__" else this.word.toLowerCase()

  /**
   * The empty filler is used to denote unspecified values, the same of the [com.kotlinnlp.conllio] library.
   */
  private val emptyFiller: String = com.kotlinnlp.conllio.Token.emptyFiller

  /**
   * @param dependencyTree a dependency tree from which to take information about this token
   * @param replacePOS a Boolean indicating whether to replace the token POS tag with the related one in the given
   *                   dependency tree (default = true)
   *
   * @return the CoNLL object that represents this token
   */
  fun toCoNLL(dependencyTree: DependencyTree? = null, replacePOS: Boolean = true) = com.kotlinnlp.conllio.Token(
    id = this.id + 1, // id starts from 1 in the CoNLL format
    form = this.word,
    lemma = this.emptyFiller,
    pos = (if (replacePOS) dependencyTree?.posTags?.get(this.id)?.label else this.pos) ?: this.emptyFiller,
    pos2 = this.emptyFiller,
    feats = emptyMap(),
    head = if (dependencyTree != null)
      dependencyTree.heads[this.id]?.plus(1) ?: 0 // the root id is 0
    else
      null,
    deprel = dependencyTree?.deprels?.get(this.id)?.label ?: this.emptyFiller,
    multiWord = this.multiWord
  )

  /**
   * @param dependencyTree a dependency tree from which to take information about this token
   *
   * @return the JSON object that represents this token
   */
  fun toJSON(dependencyTree: DependencyTree? = null): JsonObject = json {
    obj(
      "id" to this@Token.id,
      "form" to this@Token.word,
      "lemma" to null,
      "head" to dependencyTree?.heads?.get(this@Token.id),
      "pos" to dependencyTree?.posTags?.get(this@Token.id)?.label,
      "deprel" to dependencyTree?.deprels?.get(this@Token.id)?.label
    )
  }

  /**
   * @return the essential string representation of this token composed by the [id] and the [word]
   */
  override fun toString(): String = "${this.id}\t${this.word}"
}
