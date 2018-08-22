/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.token.properties.CoReference

/**
 *
 */
class CoordCorefHelper : CorefHelper {

  /**
   *
   */
  override fun setCoref(sentence: MorphoSyntacticSentence) {

    sentence.tokens.filter { it.isVerb && it.isCoord2Nd }.forEach { token ->

      sentence.getTraceSubj(token)?.let { subj ->
        sentence.getGovernor(token)?.let { governor ->

          if (governor.isVerb &&
            (token.mainMorphology as Verb).agree(governor.mainMorphology as Verb)) { // TODO: fix condition

            sentence.getSubj(governor)?.let {
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
}
