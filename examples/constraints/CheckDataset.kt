/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints

import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Number as NumberMorpho
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.morphopercolator.MorphoPercolator
import constraints.utils.ConstraintsValidator
import constraints.utils.loadConstraints
import constraints.utils.loadSentences
import java.io.File
import java.io.FileInputStream

/**
 * Read constraints from a given JSON file and verify them on a given CoNLL dataset.
 *
 * Command line arguments:
 *  1. The path of the JSON constraints file.
 *  2. The path of the CoNLL dataset file.
 *  3. The path of the morphology dictionary
 */
fun main(args: Array<String>) {

  val dictionary = run {
    println("Loading morphology dictionary from '${args[2]}'...")
    MorphologyDictionary.load(FileInputStream(File(args[2])))
  }

  ConstraintsValidator(
    constraints = loadConstraints(args[0]),
    sentences = loadSentences(filename = args[1]),
    morphoPreprocessor = MorphoPreprocessor(
      MorphologicalAnalyzer(language = dictionary.language, dictionary = dictionary)),
    morphoPercolator = MorphoPercolator(language = dictionary.language)
  ).validate()
}

