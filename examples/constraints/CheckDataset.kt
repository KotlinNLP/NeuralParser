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
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.constraints.SingleConstraint
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
  val sentences: List<CoNLLSentence> = loadSentences(filename = args[1])

  println("Verifying constraints...")

  val validationInfo: ConstraintsValidator.ValidationInfo = ConstraintsValidator(
    constraints = loadConstraints(args[0]),
    sentences = sentences,
    morphoPreprocessor = MorphoPreprocessor(
      MorphologicalAnalyzer(language = dictionary.language, dictionary = dictionary)),
    morphoPercolator = MorphoPercolator(language = dictionary.language)
  ).validate()

  println("\nValid sentences: %d / %d (%.1f%%)."
    .format(validationInfo.correctSentences, validationInfo.totalSentences, validationInfo.correctPerc))
  printConstraintsViolated(validationInfo)
}

/**
 * Print the constraints that have been violated.
 *
 * @param validationInfo info of validation
 */
private fun printConstraintsViolated(validationInfo: ConstraintsValidator.ValidationInfo) {

  if (validationInfo.violations.isNotEmpty()) {

    println("\nConstraints violated:")
    println(validationInfo.violations.keys
      .asSequence()
      .map { it.description }
      .sorted()
      .joinToString("\n") { it.prependIndent("- ") })

    println("\nPer sentence:")

    validationInfo.violations.entries.sortedBy { it.key.description }.forEach { (constraint, info) ->

      val perc: Double = 100.0 * info.violatedSentences / validationInfo.totalSentences

      println()
      println("------------------------------------------------------------------------------------------")
      println(
        "'$constraint' (violated in ${info.violatedSentences} sentences of ${validationInfo.totalSentences} [%.1f%%])"
          .format(perc))
      println("------------------------------------------------------------------------------------------")

      info.violations.forEach { tokenInfo ->
        println("\n" + buildPrintSentence(constraint = constraint, tokenInfo = tokenInfo))
      }
    }

  } else {
    println("\nNo constraints violated.")
  }
}

/**
 * @param constraint a constraint violated
 * @param tokenInfo info of the token of the violation
 *
 * @return a string containing info of the violated sentence
 */
private fun buildPrintSentence(constraint: Constraint, tokenInfo: ConstraintsValidator.TokenInfo): String {

  val governorId: Int? = tokenInfo.token.syntacticRelation.governor
  val conllLines: List<String> = tokenInfo.conllSentence.toCoNLLString(writeComments = false).split("\n")

  return if (constraint is SingleConstraint || governorId == null)
    conllLines.joinToString("\n") {
      if (Regex("^${tokenInfo.conllTokenId}\t.*").matches(it)) "$it\t<===== X" else it
    }
  else
    conllLines.joinToString("\n") {
      when {
        Regex("^${tokenInfo.conllTokenId}\t.*").matches(it) -> "$it\t<===== DEP"
        Regex("^$governorId\t.*").matches(it) -> "$it\t<===== GOV"
        else -> it
      }
    }
}
