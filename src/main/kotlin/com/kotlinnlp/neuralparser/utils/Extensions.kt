/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils

import com.kotlinnlp.conllio.CoNLLReader
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Sentence.InvalidTree
import java.io.File

/**
 * Load sentences from a CoNLL file.
 *
 * @param type the string that describes the type of sentences
 * @param filePath the file path
 * @param maxSentences the max number of sentences to load
 * @param skipNonProjective whether to skip non-projective sentences
 *
 * @return the list of loaded sentences
 */
fun loadSentences(type: String,
                  filePath: String,
                  maxSentences: Int?,
                  skipNonProjective: Boolean): List<CoNLLSentence> {

  println("Loading $type sentences from '%s'%s%s...".format(
    filePath,
    maxSentences?.let { " (max $it)" } ?: "",
    if (skipNonProjective) " skipping non-projective" else ""
  ))

  return filePath.loadFromTreeBank(skipNonProjective = skipNonProjective, maxSentences = maxSentences)
}

/**
 * Return a list of CoNLL sentences from a tree-bank at this path.
 *
 * @param maxSentences the maximum number of sentences to load (null = unlimited)
 * @param skipNonProjective whether to skip non-projective sentences
 *
 * @throws InvalidTree if the tree of a sentence is not valid
 */
private fun String.loadFromTreeBank(maxSentences: Int? = null,
                                    skipNonProjective: Boolean = false): List<CoNLLSentence> {

  var index = 0
  val sentences = ArrayList<CoNLLSentence>()

  CoNLLReader.forEachSentence(File(this)) { sentence ->

    if (maxSentences == null || index < maxSentences) {

      if (sentence.hasAnnotatedHeads()) sentence.assertValidCoNLLTree()

      val skip: Boolean = skipNonProjective && sentence.isNonProjective()

      if (!skip) sentences.add(sentence)
    }

    index++
  }

  return sentences.toList()
}

