/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package training

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.neuralparser.parsers.distance.*
import com.kotlinnlp.neuralparser.parsers.distance.decoder.PointerDistanceDecoder
import com.xenomachina.argparser.mainBody

/**
 * Train a [DistanceParser] that uses a [PointerDistanceDecoder].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  trainDistanceParser(args = args, dependencyDecoderClass = PointerDistanceDecoder::class)
}
