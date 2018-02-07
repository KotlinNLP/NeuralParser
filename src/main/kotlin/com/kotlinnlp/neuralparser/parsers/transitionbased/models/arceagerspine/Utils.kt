/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.models.arceagerspine

import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineState
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.ArcEagerSpineTransition
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.transitions.ArcLeft
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.transitions.ArcRight
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.transitions.Root
import com.kotlinnlp.syntaxdecoder.transitionsystem.models.arceagerspine.transitions.Shift

/**
 *
 */
object Utils {

  /**
   * @return the group id of this transition used by BiRNNArcEagerSpineActionsScorer and the
   * ArcEagerSpineEmbeddingsFeaturesExtractor
   */
  fun getGroupId(transition: Transition<ArcEagerSpineTransition, ArcEagerSpineState>): Int =
    when (transition) {
      is Shift -> 0
      is Root -> 1
      is ArcLeft -> 2
      is ArcRight -> 3
      else -> throw RuntimeException("invalid transition ${this}")
    }
}
