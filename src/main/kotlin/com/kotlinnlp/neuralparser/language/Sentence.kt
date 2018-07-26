/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.sentence.RealSentence
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * Data class containing the properties of a sentence.
 *
 * @property tokens the list of tokens that compose the sentence
 * @property position the position of this sentence in the original text
 */
data class Sentence(
  override val tokens: List<Token>,
  override val position: Position
) : RealSentence<Token>