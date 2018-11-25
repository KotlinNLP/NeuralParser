/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.linguisticdescription.morphology.MorphologicalAnalysis
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable

/**
 * The sentence used as input of the [com.kotlinnlp.neuralparser.NeuralParser].
 *
 * @property tokens the list of tokens of the sentence
 * @property morphoAnalysis the morphological analysis of the tokens (can be null)
 */
class ParsingSentence(
  override val tokens: List<ParsingToken>,
  override val morphoAnalysis: MorphologicalAnalysis? = null
) : MorphoSentence<ParsingToken>, SentenceIdentificable<ParsingToken>()