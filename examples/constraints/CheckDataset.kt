/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import com.kotlinnlp.conllio.CoNLLReader
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Number as NumberMorpho
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import constraints.utils.ConstraintsValidator
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
      MorphologicalAnalyzer(language = dictionary.language, dictionary = dictionary))
  ).validate()
}

/**
 * @param filename the path of the JSON file with the constraints
 *
 * @return the list of constraints read
 */
private fun loadConstraints(filename: String): List<Constraint> {

  println("Loading linguistic constraints from '$filename'...")

  return (Parser().parse(filename) as JsonArray<*>).map { Constraint(it as JsonObject) }
}

/**
 * @param filename the path of the CoNLL dataset
 *
 * @return the list of CoNLL sentences read
 */
private fun loadSentences(filename: String): List<CoNLLSentence> {

  val sentences = mutableListOf<CoNLLSentence>()

  println("Loading sentences from '$filename'...")

  CoNLLReader.forEachSentence(File(filename)) {

    require(it.hasAnnotatedHeads()) { "A dataset with annotated heads is required to check linguistic constraints." }
    it.assertValidCoNLLTree()

    sentences.add(it)
  }

  return sentences
}
