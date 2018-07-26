/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.SyntacticToken
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.ParsingSentence

/**
 * A helper that wraps a generic [NeuralParser] to let it working on CoNLL sentences.
 *
 * @property neuralParser a generic neural parser to use it with input/output sentences in CoNLL format
 */
class CoNLLDependencyParser(private val neuralParser: NeuralParser<*>) {

  /**
   * Parse a CoNLL sentence.
   *
   * @property sentence the sentence to parse, in CoNLL format
   *
   * @return the parsed sentence in CoNLL format
   */
  fun parse(sentence: CoNLLSentence): CoNLLSentence {

    val parsedSentence: MorphoSyntacticSentence = this.neuralParser.parse(ParsingSentence.fromCoNLL(sentence))

    return sentence.copy(tokens = sentence.tokens.map {

      // Note: CoNLL tokens ids start from 1, ParsingTokens ids start from 0.
      val parsedToken: SyntacticToken = parsedSentence.getTokenByID(it.id - 1)

      it.copy(
        head = parsedToken.dependencyRelation.governor?.plus(1) ?: 0, // the CoNLL root ID is 0
        deprel = parsedToken.dependencyRelation.deprel)
    })
  }
}
