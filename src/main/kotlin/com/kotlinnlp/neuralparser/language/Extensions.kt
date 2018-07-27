/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.conllio.Token as CoNLLToken

/**
 * @return a list of base real tokens
 */
fun List<CoNLLToken>.toBaseTokens(): List<BaseToken> {

  var end = -2

  return this.mapIndexed { i, it ->

    val start = end + 2 // each couple of consecutive tokens is separated by a spacing char
    end = start + it.form.length - 1

    BaseToken(
      id = it.id - 1,
      form = it.form,
      position = Position(index = i, start = start, end = end)
    )
  }
}
