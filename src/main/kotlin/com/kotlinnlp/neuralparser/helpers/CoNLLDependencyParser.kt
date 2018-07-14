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
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * @property neuralParser the neural parser
 * @property sentences the sentences to parse
 * @property verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
abstract class CoNLLDependencyParser(
  private val neuralParser: NeuralParser<*>,
  protected val sentences: List<CoNLLSentence>,
  protected val verbose: Boolean = true
) {

  /**
   * Parse the [sentences].
   *
   * @return the resulting parsed sentences in the CoNLL format
   */
  fun parse(): List<CoNLLSentence> {

    val progress: ProgressIndicatorBar? = if (this.verbose) ProgressIndicatorBar(this.sentences.size) else null

    if (this.verbose) println("Start parsing of %d sentences:".format(this.sentences.size))

    return this.sentences.map { sentence ->

      progress?.tick()

      val parsedSentence: MorphoSyntacticSentence = this.neuralParser.parse(sentence.toParsingSentence())

      CoNLLSentence(
        sentenceId = sentence.sentenceId,
        text = sentence.text,
        tokens = List(size = sentence.tokens.size, init = { i ->

          val parsedToken: SyntacticToken = parsedSentence.tokens[i] // TODO: fix index/id issue

          sentence.tokens[i].copy(
            head = parsedToken.dependencyRelation.governor?.plus(1) ?: 0, // the root id is 0
            deprel = parsedToken.dependencyRelation.deprel)
        }))
    }
  }
}
