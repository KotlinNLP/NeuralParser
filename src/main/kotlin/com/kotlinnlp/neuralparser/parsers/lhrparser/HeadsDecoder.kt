/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.ArcScores

/**
 * The interface for a latent syntactic structure decoder.
 */
interface HeadsDecoder {

  /**
   * Calculate the arc-scores.
   *
   * @param lss the latent syntactic structure to decode
   *
   * @return the computed scores
   */
  fun decode(lss: LatentSyntacticStructure): ArcScores
}