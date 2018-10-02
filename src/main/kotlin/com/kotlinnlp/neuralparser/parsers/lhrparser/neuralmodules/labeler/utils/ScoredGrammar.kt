/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration

/**
 * The outcome of a single prediction of the labeler.
 *
 * @property config the grammatical configuration
 * @property score the score
 */
data class ScoredGrammar(val config: GrammaticalConfiguration, val score: Double)
