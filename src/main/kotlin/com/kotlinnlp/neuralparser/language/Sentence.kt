/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag
import java.io.Serializable

/**
 * The Sentence.
 *
 * @param tokens a list of [Token]
 * @param dependencyTree the dependency tree associated to the [tokens] (can be null)
 */
class Sentence(val tokens: List<Token>, val dependencyTree: DependencyTree? = null) : Serializable {

  companion object {

    /**
     * Build a sentence from a CoNLL Sentence.
     *
     * @param sentence a ConLL Sentence
     *
     * @return a new sentence
     */
    fun fromCoNLL(sentence: com.kotlinnlp.conllio.Sentence) = Sentence(
      tokens = sentence.tokens.map { Token(id = it.id - 1, word = it.form, pos = it.pos, multiWord = it.multiWord) },
      dependencyTree = if (sentence.hasAnnotatedHeads()) buildDependencyTree(sentence) else null
    )

    /**
     * Build a Dependency Tree from a [sentence].
     *
     * @param sentence a sentence
     *
     * @return a new DependencyTree
     */
    private fun buildDependencyTree(sentence: com.kotlinnlp.conllio.Sentence): DependencyTree {

      val dependencyTree = DependencyTree(size = sentence.tokens.size)

      sentence.tokens.forEach { token -> dependencyTree.addArc(token) }

      return dependencyTree
    }

    /**
     * Add the arc defined by the id, head and deprel of a given [token].
     *
     * @param token a token
     */
    private fun DependencyTree.addArc(token: com.kotlinnlp.conllio.Token) {

      val id = token.id - 1
      val head = if (token.head!! == 0) null else token.head!! - 1

      val deprel = Deprel(
        label = token.deprel,
        direction = when {
          head == null -> Deprel.Position.ROOT
          head > id -> Deprel.Position.LEFT
          else -> Deprel.Position.RIGHT
        })

      if (head != null) {
        this.setArc(dependent = id, governor = head, deprel = deprel, posTag = POSTag(label = token.pos))
      } else {
        this.setDeprel(dependent = id, deprel = deprel)
        this.setPosTag(dependent = id, posTag = POSTag(label = token.pos))
      }
    }
  }

  /**
   * Convert this Sentence in a CoNLL Sentence.
   *
   * @param dependencyTree a dependency tree (optional)
   * @param replacePOS a Boolean indicating whether to replace the token POS tag with the related one in the given
   *                   dependency tree (default = true)
   *
   * @return a CoNLL Sentence
   */
  fun toCoNLL(dependencyTree: DependencyTree? = null, replacePOS: Boolean = true): com.kotlinnlp.conllio.Sentence {

    val tree = dependencyTree ?: this.dependencyTree

    return com.kotlinnlp.conllio.Sentence(
      sentenceId = "_",
      text = "_",
      tokens = this.tokens.map { it.toCoNLL(dependencyTree = tree, replacePOS = replacePOS) }.toTypedArray())
  }

  /**
   * @param dependencyTree a dependency tree from which to take information about this sentence
   *
   * @return the JSON array (of Token objects) that represents this sentence
   */
  fun toJSON(dependencyTree: DependencyTree? = null): JsonArray<JsonObject> {

    val tree = dependencyTree ?: this.dependencyTree

    return JsonArray(this@Sentence.tokens.map { it.toJSON(dependencyTree = tree) })
  }
}
