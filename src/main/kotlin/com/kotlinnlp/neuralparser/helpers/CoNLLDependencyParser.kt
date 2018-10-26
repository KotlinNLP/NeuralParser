/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.utils.notEmptyOr

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

    val parsedSentence: MorphoSynSentence = this.neuralParser.parse(
      this.sentencePreprocessor.convert(BaseSentence.fromCoNLL(sentence, index = index)))

    return sentence.copy(tokens = sentence.tokens.map {

      val parsedToken: MorphoSynToken = parsedSentence.getTokenById(it.id)

      it.copy(
        head = parsedToken.syntacticRelation.governor ?: 0, // Note: the CoNLL root ID is 0
        posList = parsedToken.flatPOS.notEmptyOr { listOf(POSTag(CoNLLToken.EMPTY_FILLER)) },
        syntacticDependencies = parsedToken.flatSyntacticRelations.map { it.dependency }
      )
    })
  }
}
