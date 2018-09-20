/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.CoReference

/**
 *
 */
class ExplicitCorefHelper : CorefHelper {

  /**
   *
   */
  override fun setCoref(sentence: MorphoSynSentence) {

    sentence.tokens.filter {
      it.isVerb && it.isRMod && it.isVerbExplicit
    }.forEach { token ->

      sentence.getTraceSubj(token)?.let { subj ->
        sentence.getGovernor(token)?.let { governor ->

          if (governor.isVerb) {

            this.selectCoref(
              verb = token,
              govSubj = sentence.getSubj(governor),
              govObj = sentence.getObj(governor))?.let {

              subj.addCoReference(CoReference(
                tokenId = it.id,
                sentenceId = sentence.id,
                confidence = 0.0))
            }
          }
        }
      }
    }
  }

  /**
   *
   */
  private fun selectCoref(verb: MorphoSynToken,
                          govSubj: MorphoSynToken?,
                          govObj: MorphoSynToken?): MorphoSynToken? {


    val govSubjAgree: Boolean = govSubj?.let {
      val govSubjMorphologies: List<SingleMorphology> = it.flatMorphologies
      verb.flatMorphologies.any { vMorpho ->
        govSubjMorphologies.any { sMorpho -> vMorpho.agree(sMorpho, weakMatch = true) }
      }
    } == true

    val govObjAgree: Boolean = govObj?.let {
      val govObjMorphologies: List<SingleMorphology> = it.flatMorphologies
      verb.flatMorphologies.any { vMorpho ->
        govObjMorphologies.any { oMorpho -> vMorpho.agree(oMorpho, weakMatch = true) }
      }
    } == true

    return if (govSubjAgree && govObjAgree) {

      govSubj // TODO: return the govSubj or the govObj based on preferential attachments

    } else if (govSubjAgree) {

      govSubj

    } else {

      null
    }
  }
}
