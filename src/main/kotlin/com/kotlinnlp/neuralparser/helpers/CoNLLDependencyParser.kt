/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.WordComposite
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence

/**
 * A helper that wraps a generic [NeuralParser] to let it working on CoNLL sentences.
 *
 * @property neuralParser a generic neural parser to use it with input/output sentences in CoNLL format
 * @property sentencePreprocessor the sentence preprocessor (e.g. to perform morphological analysis)
 */
class CoNLLDependencyParser(
  private val neuralParser: NeuralParser<*>,
  private val sentencePreprocessor: SentencePreprocessor = BasePreprocessor()
) {

  /**
   * Parse a CoNLL sentence.
   *
   * @param sentence the sentence to parse, in CoNLL format
   * @param index the index of the sentence within the list of sentences of the input dataset
   *
   * @return the parsed sentence in CoNLL format
   */
  fun parse(sentence: CoNLLSentence, index: Int): CoNLLSentence {

    val parsedSentence: MorphoSyntacticSentence = this.neuralParser.parse(
      this.sentencePreprocessor.process(BaseSentence.fromCoNLL(sentence, index = index)))

    return sentence.copy(tokens = sentence.tokens.map {

      val parsedToken: MorphoSyntacticToken = parsedSentence.getTokenById(it.id)

      it.copy(
        head = parsedToken.syntacticRelation.governor ?: 0, // the CoNLL root ID is 0
        syntacticDependencies = (parsedToken as? WordComposite)?.components?.map { it.syntacticRelation.dependency }
          ?: listOf(parsedToken.syntacticRelation.dependency)
      )
    })
  }
}
