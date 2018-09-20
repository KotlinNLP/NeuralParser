/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence

/**
 * The TracesHandler is a Deep-parsing module which inserts the tokens that are missing on the surface level but
 * important to maintain the consistency of the syntactic structure.
 */
interface TracesHandler {

  /**
   * Add the traces to the token list of the given [sentence].
   * For example, add the implicit subjects for null-subject languages.
   *
   * @param sentence the sentence
   */
  fun addTraces(sentence: MorphoSynSentence)
}
