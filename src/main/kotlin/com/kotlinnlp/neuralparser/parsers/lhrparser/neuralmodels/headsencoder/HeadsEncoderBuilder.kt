/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder

/**
 * A simple [HeadsEncoder] builder.
 *
 * @param model the encoder model
 */
class HeadsEncoderBuilder(model: HeadsEncoderModel){

  /**
   * The context encoder.
   */
  private val headsEncoder = HeadsEncoder(model)

  /**
   * @return the [headsEncoder]
   */
  operator fun invoke(): HeadsEncoder = this.headsEncoder
}
