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
import java.io.File

/**
 * @property neuralParser a neural parser
 * @property goldFilePath the path of the file containing the gold tree-bank, in CoNLL format.
 * @property outputFilePath the file path of the output CoNLL corpus (default = null -> a temporary file is used)
 * @property verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class ExternalCoNLLValidator(
  neuralParser: NeuralParser<*>,
  private val goldFilePath: String,
  private val outputFilePath: String? = null,
  verbose: Boolean
) : CoNLLDependencyParser(
  neuralParser = neuralParser,
  sentences = loadSentences(
    type = "validation",
    filePath = goldFilePath,
    maxSentences = null,
    skipNonProjective = false),
  verbose = verbose
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
   * Print the statistics resulting from the official CoNLL evaluation script.
   *
   * @return the statistics of the evaluation
   */
  fun evaluate() {

    val parsedSentences: List<CoNLLSentence> = this.parse()

    print("\nCoNLL official script evaluation:\n%s".format(this.evaluateWithCoNLLScript(parsedSentences)))
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
