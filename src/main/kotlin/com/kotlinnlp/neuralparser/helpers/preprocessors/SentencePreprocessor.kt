/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.preprocessors

import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.BaseToken
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.tokensencoder.wrapper.SentenceConverter

/**
 * Pre-process a sentence before the parsing starts.
 */
interface SentencePreprocessor : SentenceConverter<BaseToken, BaseSentence, ParsingToken, ParsingSentence>

