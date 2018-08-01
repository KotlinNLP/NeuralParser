/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

import com.kotlinnlp.linguisticdescription.language.Language
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.BasePreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import java.io.File
import java.io.FileInputStream

/**
 * Build a [SentencePreprocessor].
 *
 * @param morphoDictionaryPath the path of the serialized morphology dictionary
 * @param language the language in which to process the sentence
 *
 * @return a new sentence preprocessor
 */
internal fun buildSentencePreproessor(morphoDictionaryPath: String?, language: Language): SentencePreprocessor {

  return morphoDictionaryPath?.let {

    println("Loading serialized dictionary from '$it'...")

    MorphoPreprocessor(
      MorphologicalAnalyzer(language = language, dictionary = MorphologyDictionary.load(FileInputStream(File(it))))
    )

  } ?: BasePreprocessor()
}
