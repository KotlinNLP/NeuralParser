/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils

import com.kotlinnlp.dependencytree.Deprel

/**
 * The outcome of a single prediction of the labeler.
 *
 * @property value the deprel
 * @property score the score
 */
data class ScoredDeprel(val value: Deprel, val score: Double) {

  companion object {

    /**
     * A ScoredDeprel [Comparator] by score.
     */
    val deprelComparator = compareByDescending<ScoredDeprel> { it.score }
  }
}

/**
 * List of ScoredDeprel,
 */
typealias ScoredDeprelList = List<ScoredDeprel>

/**
 * @return a sorted the list of prediction by descending score
 */
fun ScoredDeprelList.sortByScore() = this.sortedWith(ScoredDeprel.deprelComparator)

