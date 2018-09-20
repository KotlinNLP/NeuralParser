/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser

import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.neuralparser.language.ParsingSentence

/**
 * A Neural Parser.
 */
interface NeuralParser<out ModelType: NeuralParserModel> {

  /**
   * The model of this neural parser.
   */
  val model: ModelType

  /**
   * Parse a sentence, giving its dependency tree.
   *
   * @param sentence a [Sentence]
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  fun parse(sentence: ParsingSentence): MorphoSynSentence
}
