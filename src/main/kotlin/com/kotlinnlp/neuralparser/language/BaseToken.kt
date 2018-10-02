/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * A base real token.
 *
 * @property id the token id, an incremental integer starting from 0 within a sentence
 * @property form the form of the token
 * @property position the position of the token in the original text
 */
data class BaseToken(
  override val id: Int,
  override val form: String,
  override val position: Position
) : RealToken, TokenIdentificable
