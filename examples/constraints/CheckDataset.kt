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
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.constraints.SingleConstraint
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import constraints.utils.explodeTokens
import constraints.utils.toMorphoSyntactic
import java.io.File

/**
 * Read constraints from a given JSON file and verify them on a given CoNLL dataset.
 *
 * Command line arguments:
 *  1. The path of the JSON constraints file.
 *  2. The path of the CoNLL dataset file.
 */
fun main(args: Array<String>) {

  Validator(
    constraints = loadConstraints(args[0]),
    sentences = loadSentences(args[1])
  ).validate()
}

/**
 * @param filename the path of the JSON file with the constraints
 *
 * @return the list of constraints read
 */
private fun loadConstraints(filename: String): List<Constraint> {

  println("Loading linguistic constraints from '$filename'")

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
 */
private class Validator(private val constraints: List<Constraint>, private val sentences: List<CoNLLSentence>) {

  /**
   * The map of constraint descriptions associated to the lists of related sentences.
   */
  private val constraintsViolated = mutableMapOf<String, MutableList<String>>()

  /**
   * The count of correct sentences
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

    var isValid = true
    val sentenceTree = DependencyTree(sentence)
    var nextAvailableId: Int = sentence.tokens.last().id + 1
    val tokens: List<MorphoSynToken> = sentence.tokens.map {
      val morphoSynToken = it.toMorphoSyntactic(tree = sentenceTree, nextAvailableId = nextAvailableId)
      if (morphoSynToken is MorphoSynToken.Composite) nextAvailableId += morphoSynToken.components.size
      morphoSynToken
    }
    val explodedTokens: List<MorphoSynToken.Single> = explodeTokens(tokens)
    val tokensTree = DependencyTree(explodedTokens)

    this.constraints.forEach { constraint ->
      explodedTokens.forEach { token ->

        if (!constraint.isVerified(token = token, tokens = explodedTokens, dependencyTree = tokensTree)) {

          this.constraintsViolated.getOrPut(constraint.description) { mutableListOf() }
            .add(this.buildPrintSentence(constraint = constraint, sentence = sentence, token = token))

          isValid = false
        }
      }
    }

    if (isValid) this.correct++
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

  /**
   * Print the constraints that have been violated.
   */
  private fun printConstraintsViolated() {

    if (this.constraintsViolated.isNotEmpty()) {

      println("\nConstraints violated:")

      this.constraintsViolated.forEach { constraint, sentences ->

        val perc: Double = 100.0 * sentences.size / this.sentences.size

        println()
        println("------------------------------------------------------------------------------------------")
        println("'$constraint' (violated in ${sentences.size} sentences of ${this.sentences.size} [%.1f%%])"
          .format(perc))
        println("------------------------------------------------------------------------------------------")

        sentences.forEach { println("\n" + it) }
      }

    } else {
      println("\nNo constraints violated.")
    }
  }
}
