/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.dependencytree.Deprel
import com.kotlinnlp.dependencytree.POSTag

/**
 * A dependency relation composed by a POS tag and a deprel.
 *
 * @property posTag the POS tag
 * @property deprel the deprel
 */
data class DependencyRelation(val posTag: POSTag, val deprel: Deprel)
