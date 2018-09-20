/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.CoReference

/**
 *
 */
class InfinitiveCorefHelper : CorefHelper {

  /**
   * @property SUBJ
   * @property OBJ
   * @property IOBJ
   * @property UNDEFINED
   */
  enum class TraceControl { SUBJ, OBJ, IOBJ, UNDEFINED }

  /**
   * @property subj
   * @property obj
   * @property iobj
   */
  data class CorefCandidates(
    val subj: MorphoSynToken?,
    val obj: MorphoSynToken?,
    val iobj: MorphoSynToken?
  ) {

    /**
     * @param control
     */
    fun select(control: TraceControl): MorphoSynToken? = when {
      this.iobj != null && control == TraceControl.IOBJ -> this.iobj
      this.obj != null && control == TraceControl.OBJ -> this.obj
      this.subj != null && control == TraceControl.SUBJ -> this.subj
      else -> this.subj // the most probable option
    }
  }

  /**
   * Set the co-references in the [sentence] tokens.
   *
   * @param sentence the sentence
   */
  override fun setCoref(sentence: MorphoSyntacticSentence) {

    sentence.tokens.filter { it.isVerbNotAux && it.isVerbInfinite }.forEach { token ->

      sentence.getTraceSubj(token)?.let { subj ->
        this.findSubjCoref(sentence, token)?.let { coRef ->
          subj.addCoReference(coRef)
        }
      }
    }
  }

  /**
   * @param sentence the sentence
   * @param token a token
   */
  private fun findSubjCoref(sentence: MorphoSyntacticSentence,
                            token: MorphoSynToken): CoReference? {

    return sentence.getGovernor(token)?.let { governor ->

      when {

        governor.isVerb -> this.findCandidates(sentence, governor).select(control = this.getControl(governor))?.let {

          CoReference(tokenId = it.id,
            sentenceId = sentence.id,
            confidence = 0.0) // TODO: set confidence
        }

        // governor.isNoun -> TODO()

        else -> TODO()
      }
    }
  }

  /**
   *
   */
  private fun findCandidates(sentence: MorphoSyntacticSentence,
                             token: MorphoSynToken): CorefCandidates = with(sentence) {

    CorefCandidates(
      subj = getSubj(token),
      obj = getObj(token),
      iobj = getIObj(token)
    )
  }

  /**
   *
   */
  private fun getControl(token: MorphoSynToken): TraceControl = TODO()
}
