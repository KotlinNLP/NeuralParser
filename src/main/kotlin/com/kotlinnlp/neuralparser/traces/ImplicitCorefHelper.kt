/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.traces

import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.linguisticdescription.sentence.token.properties.CoReference

/**
 *
 */
class ImplicitCorefHelper : CorefHelper {

  /**
   *
   */
  override fun setCoref(sentence: MorphoSynSentence) {

    sentence.tokens.filter {
      it.isVerb && it.isRMod && (it.isVerbGerundive || it.isVerbParticiple)
    }.forEach { token ->

      sentence.getTraceSubj(token)?.let { subj ->
        sentence.getGovernor(token)?.let { governor ->

          if (governor.isVerb) {

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
