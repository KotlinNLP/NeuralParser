/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels

import com.kotlinnlp.simplednn.core.optimizer.Optimizer

/**
 * Generic NeuralModel Interface.
 */
interface NeuralModel<
  in InputType : Any,
  out OutputType : Any,
  in ErrorsType : Any,
  out InputErrorsType : Any,
  out ParamsErrorsType : Any
  > {

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  fun forward(input: InputType): OutputType

  /**
   * The Backward.
   *
   * @param errors the errors of the last forward
   */
  fun backward(errors: ErrorsType)

  /**
   * Return the input errors of the last backward.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  fun getInputErrors(copy: Boolean = true): InputErrorsType

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  fun getParamsErrors(copy: Boolean = true): ParamsErrorsType

  /**
   * Back-propagate the [errors] through the network, accumulate the resulting params errors in the
   * [optimizer] and returns the input errors.
   *
   * @param errors the output errors
   * @param optimizer the optimizer
   * @param copy a Boolean indicating whether the errors must be a copy or a reference (default false)
   *
   * @return the input errors
   */
  fun propagateErrors(errors: ErrorsType, optimizer: Optimizer<ParamsErrorsType>, copy: Boolean = false): InputErrorsType {

    backward(errors)
    optimizer.accumulate(getParamsErrors(copy = copy))
    return getInputErrors(copy = copy)
  }
}
