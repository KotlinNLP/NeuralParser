/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Noun
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSyntacticToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.CoReference

/**
 *
 */
class ExplicitCorefHelper : CorefHelper {

  /**
   *
   */
  override fun setCoref(sentence: MorphoSyntacticSentence) {

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
  private fun selectCoref(verb: MorphoSyntacticToken,
                          govSubj: MorphoSyntacticToken?,
                          govObj: MorphoSyntacticToken?): MorphoSyntacticToken? {


    val govSubjAgree: Boolean = govSubj?.let {
      SingleMorphology.agree(
        first = it.mainMorphology as Noun, // TODO: fix cast
        second = verb.mainMorphology as Verb,
        weakMatch = true)
    } == true

    val govObjAgree: Boolean = govObj?.let {
      SingleMorphology.agree(
        first = it.mainMorphology as Noun, // TODO: fix cast
        second = verb.mainMorphology as Verb,
        weakMatch = true)
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
