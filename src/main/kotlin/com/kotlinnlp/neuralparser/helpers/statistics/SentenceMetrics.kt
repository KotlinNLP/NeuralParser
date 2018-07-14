/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.statistics

/**
 * The metrics of a sentence.
 *
 * @property correctLabeled if the parsed sentence has all correct attachments, including deprel labels
 * @property correctUnlabeled if the parsed sentence has all correct attachments, excluding deprel labels
 * @property correctLabeledNoPunct same as [correctLabeled], without considering the punctuation tokens
 * @property correctUnlabeledNoPunct same as [correctUnlabeled], without considering the punctuation tokens
 */
internal data class SentenceMetrics(
  var correctLabeled: Boolean = true,
  var correctUnlabeled: Boolean = true,
  var correctLabeledNoPunct: Boolean = true,
  var correctUnlabeledNoPunct: Boolean = true
)
