/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder

import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters

/**
 * @property biRNNParameters the params of the BiRNN of the [HeadsEncoder]
 */
class HeadsEncoderParams(val biRNNParameters: BiRNNParameters)