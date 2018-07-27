/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.preprocessors

import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.BaseSentence

/**
 * Pre-process a sentence before the parsing starts.
 */
interface SentencePreprocessor {

  /**
   * Convert a [BaseSentence] to a [ParsingSentence].
   *
   * @param sentence a base sentence
   *
   * @return a sentence ready to be parsed
   */
  fun process(sentence: BaseSentence): ParsingSentence
}
