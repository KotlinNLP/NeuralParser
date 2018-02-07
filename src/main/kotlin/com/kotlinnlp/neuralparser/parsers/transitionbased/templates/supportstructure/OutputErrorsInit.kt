/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.transitionbased.templates.supportstructure

/**
 * An enumerate used to choose the default initialization of the output errors of a processor.
 *
 * @property AllZeros outcomes errors are all initialized to zeros
 * @property AllErrors all outcomes are considered as errors by default
 */
enum class OutputErrorsInit {
  AllZeros,
  AllErrors
}
