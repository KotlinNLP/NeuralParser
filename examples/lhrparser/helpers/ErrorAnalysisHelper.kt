/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package lhrparser.helpers

import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.neuralparser.language.Sentence

/**
 * The helper that performs some error analysis.
 */
class ErrorAnalysisHelper(val parser: LHRParser) {

  /**
   * The calculated scores.
   */
  private val deprelsScores = List(
    size = this.parser.model.corpusDictionary.deprelTags.getElements().size,
    init = { mutableListOf<Double>() }
  )

  /**
   * Calculate some statistics.
   *
   * @param sentence the sentence
   * @param dependencyTree the dependency tree of the [sentence]
   */
  fun calcStats(sentence: Sentence, dependencyTree: DependencyTree) {

    dependencyTree.elements.forEach { tokenId ->

      val deprel: Deprel = dependencyTree.deprels[tokenId]!!
      val deprelId: Int = this.parser.model.corpusDictionary.deprelTags.getId(deprel)!!
      val score: Double = dependencyTree.attachmentScores[tokenId]

      this.deprelsScores[deprelId].add(score)
    }
  }

  /**
   * @return the statistics
   */
  fun getStats(): List<Pair<Deprel, Pair<Double, Pair<Double, Double>>?>> {

    val similarityAvg: List<Double?> = this.deprelsScores.map { if (it.isNotEmpty()) it.average() else null }

    val similarityStd: List<Pair<Double, Double>?> = this.deprelsScores.zip(similarityAvg)
      .map { (scores, avg) ->

        if (scores.isNotEmpty()) { avg!!

          val overScores = scores.filter { it > avg }
          val underScores = scores.filter { it < avg }

          Pair(
            if (overScores.size > 1) overScores.sumByDouble { it - avg } / overScores.size else 0.0,
            if (underScores.size > 1) underScores.sumByDouble { avg - it } / underScores.size else 0.0
          )
        } else {
          null
        }
      }

    return List(
      size = this.deprelsScores.size,
      init = { deprelId ->
        val avg = similarityAvg[deprelId]
        Pair(
          this.parser.model.corpusDictionary.deprelTags.getElement(deprelId)!!,
          if (avg != null) Pair(avg, similarityStd[deprelId]!!) else null
        )
      }
    )
  }
}