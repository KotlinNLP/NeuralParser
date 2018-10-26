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
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.POS
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Number as NumberMorpho
import com.kotlinnlp.linguisticdescription.morphology.properties.Gender
import com.kotlinnlp.linguisticdescription.morphology.properties.Number
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.constraints.SingleConstraint
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.selector.CompositeSelector
import com.kotlinnlp.utils.notEmptyOr
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import constraints.utils.explodeTokens
import constraints.utils.toMorphoSyntactic
import constraints.utils.ConstraintsValidator
import java.io.File
import java.io.FileInputStream
import java.lang.IllegalArgumentException

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

  Validator(
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

/**
 * A helper that verifies constraints on sentences.
 *
 * @param constraints the list of constraints to verify
 * @param sentences the list of sentence to validate
 * @param morphoPreprocessor a morphological sentence preprocessor
 */
private class Validator(
  private val constraints: List<Constraint>,
  private val sentences: List<CoNLLSentence>,
  private val morphoPreprocessor: MorphoPreprocessor
) {

  /**
   * Information about a constraint that has been violated.
   *
   * @property violations the list of violations as pair of <token, sentence>
   * @property violatedSentences the count of sentences in which the constraint has been violated
   */
  private class ViolationInfo(
    val violations: MutableList<Pair<MorphoSynToken.Single, CoNLLSentence>> = mutableListOf(),
    var violatedSentences: Int = 0
  )

  /**
   * The map of violated constraint to the related info.
   */
  private val violationsByConstraint = mutableMapOf<Constraint, ViolationInfo>()

  /**
   * The count of correct sentences.
   */
  private var correct = 0

  /**
   * Verify the [constraints] on the [sentences].
   */
  fun validate() {

    val progress = ProgressIndicatorBar(this.sentences.size)

    println("Verifying constraints...")

    this.sentences.forEach {
      progress.tick()
      this.processSentence(it)
    }

    println("\nValid sentences: %d / %d (%.1f%%).".format(correct, sentences.size, 100.0 * correct / sentences.size))
    this.printConstraintsViolated()
  }

  /**
   * @param sentence a sentence to process
   */
  private fun processSentence(sentence: CoNLLSentence) {

    val tokens: List<MorphoSynToken.Single> = this.buildMorphoSynTokens(sentence)
    val validator = ConstraintsValidator(constraints = this.constraints, tokens = tokens)
    val violated: Map<MorphoSynToken.Single, Set<Constraint>> = validator.validate()

    if (violated.isEmpty()) this.correct++

    violated.values.flatMap { it }.toSet().forEach {
      this.violationsByConstraint.getOrPut(it) { ViolationInfo() }.violatedSentences++
    }

    violated.forEach { token, constraints ->
      constraints.forEach { this.violationsByConstraint.getValue(it).violations.add(token to sentence) }
    }
  }

  /**
   * @param sentence a CoNLL sentence
   *
   * @return a list of morpho-syntactic tokens built from the given sentence
   */
  private fun buildMorphoSynTokens(sentence: CoNLLSentence): List<MorphoSynToken.Single> {

    val sentenceTree = DependencyTree(sentence)
    var nextAvailableId: Int = sentence.tokens.last().id + 1

    val parsingSentence: ParsingSentence = this.morphoPreprocessor.convert(BaseSentence.fromCoNLL(sentence, index = 0))

    val morphoSynTokens: List<MorphoSynToken> = sentence.tokens.mapIndexed { i, conllToken ->

      val morphoSynToken = conllToken.toMorphoSyntactic(
        tree = sentenceTree,
        nextAvailableId = nextAvailableId,
        morphologies = this.getValidMorphologies(
          configuration = sentenceTree.getConfiguration(conllToken.id)!!,
          conllToken = conllToken,
          tokenIndex = i,
          parsingSentence = parsingSentence))

      if (morphoSynToken is MorphoSynToken.Composite) nextAvailableId += morphoSynToken.components.size

      morphoSynToken
    }

    return explodeTokens(morphoSynTokens)
  }

  /**
   * Get the valid morphologies of a token respect to a given grammatical configuration.
   *
   * @param configuration a grammatical configuration of a token
   * @param conllToken the token
   * @param tokenIndex the index of the token within the sentence tokens
   * @param parsingSentence the parsing sentence of the token
   *
   * @return the list of valid morphologies of the token
   */
  private fun getValidMorphologies(configuration: GrammaticalConfiguration,
                                   conllToken: CoNLLToken,
                                   tokenIndex: Int,
                                   parsingSentence: ParsingSentence): List<Morphology> = try {

    CompositeSelector
      .getValidMorphologies(configuration = configuration, sentence = parsingSentence, tokenIndex = tokenIndex)
      .notEmptyOr {
        listOf(
          Morphology(components = configuration.components.map {
            SingleMorphology(
              lemma = conllToken.form,
              pos = (it.pos as POSTag.Base).type,
              allowIncompleteProperties = true)
          }))
      }

  } catch (e: IllegalArgumentException) {

    val pos: POS = (conllToken.posList.single() as POSTag.Base).type

    if (pos == POS.Num) // force numbers in case the morphological analyzer did not find them
      listOf(Morphology(NumberMorpho(
        lemma = conllToken.form,
        numericForm = 0,
        gender = Gender.Undefined,
        number = Number.Undefined)))
    else
      throw e
  }

  /**
   * Print the constraints that have been violated.
   */
  private fun printConstraintsViolated() {

    if (this.violationsByConstraint.isNotEmpty()) {

      println("\nConstraints violated:")

      this.violationsByConstraint.forEach { constraint, info ->

        val perc: Double = 100.0 * info.violatedSentences / this.sentences.size

        println()
        println("------------------------------------------------------------------------------------------")
        println("'$constraint' (violated in ${info.violatedSentences} sentences of ${this.sentences.size} [%.1f%%])"
          .format(perc))
        println("------------------------------------------------------------------------------------------")

        info.violations.forEach {
          println("\n" + buildPrintSentence(constraint = constraint, token = it.first, sentence = it.second))
        }
      }

    } else {
      println("\nNo constraints violated.")
    }
  }

  /**
   * @param constraint a constraint violated in the [sentence]
   * @param sentence a sentence
   * @param token the token of the [sentence] that violates the [constraint]
   *
   * @return the [sentence] to be printed
   */
  private fun buildPrintSentence(constraint: Constraint, sentence: CoNLLSentence, token: MorphoSynToken): String {

    val governorId: Int? = token.syntacticRelation.governor
    val conllSentences: List<String> = sentence.toCoNLLString(writeComments = false).split("\n")

    return if (constraint is SingleConstraint || governorId == null)
      conllSentences.joinToString("\n") {
        if (Regex("^${token.id}\t.*").matches(it)) "$it\t<===== X" else it
      }
    else
      conllSentences.joinToString("\n") {
        when {
          Regex("^${token.id}\t.*").matches(it) -> "$it\t<===== DEP"
          Regex("^$governorId\t.*").matches(it) -> "$it\t<===== GOV"
          else -> it
        }
      }
  }
}
