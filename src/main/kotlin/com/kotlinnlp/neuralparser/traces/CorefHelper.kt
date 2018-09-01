package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence

/**
 *
 */
interface CorefHelper {

  /**
   * Set the co-references in the [sentence] tokens.
   *
   * @param sentence the sentence
   */
  fun setCoref(sentence: MorphoSyntacticSentence)
}