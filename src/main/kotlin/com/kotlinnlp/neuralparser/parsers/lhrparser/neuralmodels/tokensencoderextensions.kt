/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder
import com.kotlinnlp.tokensencoder.TokensEncoderOptimizer

/**
 * Propagate the [errors] through the tokens encoder, accumulate the resulting params errors in the
 * [optimizer] and returns the input errors.
 *
 * @param errors the output errors
 * @param optimizer the optimizer
 * @param copy a Boolean indicating whether the errors must be a copy or a reference (default false)
 */
fun TokensEncoder.propagateErrors(errors: List<DenseNDArray>,
                                  optimizer: TokensEncoderOptimizer,
                                  copy: Boolean = false) = with(this) {
  backward(errors)
  optimizer.accumulate(getParamsErrors(copy = copy))
}