/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.pointerparser

/*
TODO: move to KotlinNLP/Utils

Yoshua Bengio:

My preferred style of moving average is the following. Let's say you
have a series x_t and you want to estimate the mean m of previous
(recent) x's:

m <-- m + (2/t) (x_t - m)

Note that with (1/t) learning rate instead of (2/t) you get the exact
historical average. With a larger learning rate (like 2/t) you give
a bit more importance to recent stuff, which makes sense if x's are
non-stationary (very likely here [in the setting of computing the
moving average of the training error]). With a constant learning rate
(independent of t) you get an exponential moving average.

You can estimate a running average of the gradient variance by running
averages of the mean gradient and of the
square of the difference to the moving mean.
*/
class MovingAverage {

  /**
   * The mean.
   */
  var mean: Double = 0.0
    private set

  /**
   * The variance.
   */
  var variance: Double = 0.0
    private set

  /**
   * Counts the added values.
   */
  var count: Long = 0
    private set

  /**
   * Add the given [value] to the moving average
   *
   * @param value the value to add to the moving average
   */
  fun add(value: Double) {

    this.count++
    this.mean += (2.0 / this.count) * (value - this.mean)
    this.variance += (2.0 / this.count) * ((value - this.mean) * (value - this.mean) - this.variance)
  }
}