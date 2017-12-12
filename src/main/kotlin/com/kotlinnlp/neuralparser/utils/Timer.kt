/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.utils

/**
 * A timer to track the elapsed time.
 */
class Timer {

  /**
   * When the timer started.
   */
  private var startTime: Long = System.currentTimeMillis()

  /**
   * Reset the start time.
   */
  fun reset() {
    this.startTime = System.currentTimeMillis()
  }

  /**
   * @return the milliseconds elapsed from the start time
   */
  fun getElapsedMillis(): Long = System.currentTimeMillis() - this.startTime

  /**
   * @return a string formatted with the time elapsed from the start, in seconds and minutes
   */
  fun formatElapsedTime(): String {

    val elapsedSecs = this.getElapsedMillis() / 1000.0

    return "%.3f s (%.1f min)".format(elapsedSecs, elapsedSecs / 60.0)
  }
}
