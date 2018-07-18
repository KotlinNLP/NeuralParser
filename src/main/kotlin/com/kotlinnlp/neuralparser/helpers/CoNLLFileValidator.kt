/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.conllio.CoNLLUEvaluator
import com.kotlinnlp.conllio.CoNLLWriter
import com.kotlinnlp.conllio.CoNLLXEvaluator
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.utils.loadSentences
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar
import java.io.File

/**
 * Validate a system output CoNLL file comparing it to a gold CoNLL file.
 *
 * @param neuralParser a neural parser
 * @param goldFilePath the path of the file containing the gold tree-bank, in CoNLL format.
 * @param outputFilePath the file path of the output CoNLL corpus (default = null -> a temporary file is used)
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class CoNLLFileValidator(
  neuralParser: NeuralParser<*>,
  private val goldFilePath: String,
  private val outputFilePath: String? = null,
  private val verbose: Boolean = true
) {

  /**
   * Return a temporary file absolute path.
   *
   * @return the path of a temporary file generated at runtime
   */
  private val defaultOutputPath: String get() = File.createTempFile("${this.goldFilePath}_output", ".conll").path

  /**
   * The CoNLL Evaluator
   */
  private val conllEvaluator = if (this.goldFilePath.endsWith(".conllu")) CoNLLUEvaluator else CoNLLXEvaluator

  /**
   * The parser wrapper to parse sentences in CoNLL format.
   */
  private val conllParser = CoNLLDependencyParser(neuralParser = neuralParser)

  /**
   * Print the statistics resulting from the official CoNLL evaluation script.
   *
   * @return the statistics of the evaluation
   */
  fun evaluate() {

    val parsedSentences: List<CoNLLSentence> = this.parseSentences(sentences = loadSentences(
      type = "validation",
      filePath = goldFilePath,
      maxSentences = null,
      skipNonProjective = false))

    print("\nCoNLL official script evaluation:\n%s".format(this.evaluateWithCoNLLScript(parsedSentences)))
  }

  /**
   * Parse the validation CoNLL sentences.
   *
   * @return the list of parsed CoNLL sentences
   */
  private fun parseSentences(sentences: List<CoNLLSentence>): List<CoNLLSentence> {

    val progress: ProgressIndicatorBar? = if (this.verbose) ProgressIndicatorBar(sentences.size) else null

    if (this.verbose) println("Start parsing of %d sentences:".format(sentences.size))

    return sentences.map {

      progress?.tick()

      this.conllParser.parse(it)
    }
  }

  /**
   * Get the output of the official CoNLL evaluation script.
   *
   * @param parsedSentences a list of parsed sentences, parallel to the gold sentences
   *
   * @return the output of the official CoNLL evaluation script
   */
  private fun evaluateWithCoNLLScript(parsedSentences: List<CoNLLSentence>): String? {

    val outputPath: String = this.outputFilePath ?: this.defaultOutputPath

    CoNLLWriter.toFile(sentences = parsedSentences, writeComments = true, outputFilePath = outputPath)

    return this.conllEvaluator.evaluate(systemFilePath = outputPath, goldFilePath = this.goldFilePath)
  }
}
