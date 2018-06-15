/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils

import com.kotlinnlp.conllio.CoNLLReader
import com.kotlinnlp.conllio.Sentence.InvalidTree
import com.kotlinnlp.neuralparser.language.Sentence
import java.io.File

/**
 * Populate an array of sentences from a tree-bank.
 *
 * @param filePath the file path of the tree-bank in CoNLL format
 * @param maxSentences the maximum number of sentences to load (null = unlimited)
 * @param skipNonProjective whether to skip non-projective sentences
 *
 * @throws InvalidTree if the tree of a sentence is not valid
 */
fun ArrayList<Sentence>.loadFromTreeBank(filePath: String,
                                         maxSentences: Int? = null,
                                         skipNonProjective: Boolean = false) {

  var index = 0

  CoNLLReader.forEachSentence(File(filePath)) { sentence ->

    if (maxSentences == null || index < maxSentences) {

      if (sentence.hasAnnotatedHeads()) sentence.assertValidCoNLLTree()

      val skip: Boolean = skipNonProjective && sentence.isNonProjective()

      if (!skip) this.add(Sentence.fromCoNLL(sentence))
    }

    index++
  }
}
