package com.kotlinnlp.neuralparser.helpers.preprocessors

import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken

/**
 * Pre-process a sentence before the parsing starts.
 */
class BasePreprocessor : SentencePreprocessor {

  /**
   * Convert a [BaseSentence] to a [ParsingSentence].
   *
   * @param sentence a base sentence
   *
   * @return a sentence ready to be parsed
   */
  override fun process(sentence: BaseSentence) = ParsingSentence(
    tokens = sentence.tokens.map {
      ParsingToken(id = it.id, form = it.form, position = it.position, morphologies = emptyList(), posTag = it.posTag)
    }
  )
}

