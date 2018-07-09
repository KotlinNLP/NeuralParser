/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.utils

/**
 * The available loss criterion.
 *
 * @property Softmax calculate the errors with cross-entropy softmax
 * @property HingeLoss calculate the errors with the hinge loss method
 */
enum class LossCriterionType { Softmax, HingeLoss }