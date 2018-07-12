/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.linguisticdescription.morphology.MorphologyEntry
import com.kotlinnlp.linguisticdescription.sentence.token.RealToken
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position

/**
 * @param id the token id
 * @param form the form
 * @param position the position of the token in its sentence
 * @param lexicalForms the list of morphological interpretations of the token
 */
data class ParsingToken(
  val id: Int,
  override val form: String,
  override val position: Position,
  val lexicalForms: List<MorphologyEntry> = emptyList()) : RealToken