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
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.neuralparser.utils.loadFromTreeBank
import com.kotlinnlp.progressindicator.ProgressIndicatorBar
import java.io.File

/**
 * The Validator.
 *
 * @property neuralParser a neural parser
 * @property goldFilePath the path of the file containing the gold tree-bank, in CoNLL format.
 * @property outputFilePath the file path of the output CoNLL corpus (default = null -> a temporary file is used)
 * @property verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class Validator(
  private val neuralParser: NeuralParser<*>,
  private val goldFilePath: String,
  private val outputFilePath: String? = null,
  private val verbose: Boolean = true
) {

  /**
   * A counter of statistic metrics.
   *
   * @property labeledAttachments the counter of labeled attachments
   * @property unlabeledAttachments the counter of unlabeled attachments
   * @property correctPOSTags the counter of correct POS tags
   * @property correctDeprels the counter of correct deprels
   * @property correctLabeledSentences the counter of correct labeled sentences
   * @property correctUnlabeledSentences the counter of correct unlabeled sentences
   * @property totalSentences the total amount of sentences
   * @property totalTokens the total amount of tokens
   */
  private data class MetricsCounter(
    var labeledAttachments: Int = 0,
    var unlabeledAttachments: Int = 0,
    var correctPOSTags: Int = 0,
    var correctDeprels: Int = 0,
    var correctLabeledSentences: Int = 0,
    var correctUnlabeledSentences: Int = 0,
    var totalSentences: Int = 0,
    var totalTokens: Int = 0)

  /**
   * The metrics of a sentence.
   *
   * @property correctLabeled if the parsed sentence has all correct attachments, including deprel labels
   * @property correctUnlabeled if the parsed sentence has all correct attachments, excluding deprel labels
   * @property correctLabeledNoPunct same as [correctLabeled], without considering the punctuation tokens
   * @property correctUnlabeledNoPunct same as [correctUnlabeled], without considering the punctuation tokens
   */
  private data class SentenceMetrics(
    var correctLabeled: Boolean = true,
    var correctUnlabeled: Boolean = true,
    var correctLabeledNoPunct: Boolean = true,
    var correctUnlabeledNoPunct: Boolean = true
  )

  /**
   * The list of gold [Sentence]s to evaluate.
   */
  private val goldSentences = ArrayList<Sentence>()

  /**
   * The CoNLL Evaluator
   */
  private val conllEvaluator = if (this.goldFilePath.endsWith(".conllu")) CoNLLUEvaluator else CoNLLXEvaluator

  /**
   * The regular expression to match punctuation forms.
   */
  private val punctuationRegex = Regex("^[-!\"#%&'()*,./:;?@\\[\\]_{}]+$")

  /**
   * A counter of statistic metrics.
   */
  private lateinit var counter: MetricsCounter

  /**
   * A counter of statistic metrics, without considering punctuation.
   */
  private lateinit var counterNoPunct: MetricsCounter

  /**
   * The metrics of a sentence.
   */
  private lateinit var sentenceMetrics: SentenceMetrics

  /**
   * Load the gold tree-bank from the given file.
   */
  init {
    if (this.verbose) print("Loading validation sentences... ")
    this.goldSentences.loadFromTreeBank(this.goldFilePath)
    if (this.verbose) println("Done.")
  }

  /**
   * Evaluate the gold sentences (the output of the parsed sentences is written in CoNLL format to the file at
   * [outputFilePath]).
   *
   * @param printCoNLLScriptEvaluation whether to print the output string of the official CoNLL evaluation script
   * @param afterParsing the callback to call after having analysed a sentence (can be null)
   *
   * @return the output of the evaluation script of the CoNLL
   */
  fun evaluate(printCoNLLScriptEvaluation: Boolean = false,
               afterParsing: ((sentence: Sentence, dependencyTree: DependencyTree) -> Unit)? = null): Statistics {

    val progress: ProgressIndicatorBar? = if (this.verbose) ProgressIndicatorBar(this.goldSentences.size) else null

    if (this.verbose) println("Start validation on %d sentences:".format(this.goldSentences.size))

    val dependencyTrees: List<DependencyTree> = this.goldSentences.map {

      progress?.tick()

      val result: DependencyTree = this.neuralParser.parse(it)

      afterParsing?.invoke(it, result)

      result
    }

    if (printCoNLLScriptEvaluation) {
      print("\nCoNLL official script evaluation:\n%s".format(this.evaluateWithCoNLLScript(dependencyTrees)))
    }

    return this.getStats(dependencyTrees)
  }

  /**
   * Generate a temporary file and return its absolute path.
   *
   * @return the path of a temporary file generated at runtime
   */
  private fun getDefaultOutputPath(): String
    = File.createTempFile("neural_parser_validation_output", ".conll").path

  /**
   * Get statistics about the evaluation of a given list of [dependencyTrees], made comparing them with the
   * [goldSentences].
   *
   * @param dependencyTrees a list of dependency trees of parsed sentences, parallel to the gold sentences
   *
   * @return the statistics of the evaluation of the given [dependencyTrees]
   */
  private fun getStats(dependencyTrees: List<DependencyTree>): Statistics {

    this.initCounters(dependencyTrees)

    dependencyTrees.zip(this.goldSentences).forEach { (tree, goldSentence) ->

      val goldTree: DependencyTree = goldSentence.dependencyTree!!
      require(tree.size == goldTree.size) { "The dependency tree and its gold haven't the same size" }

      this.sentenceMetrics = SentenceMetrics()

      (0 until tree.size).forEach { i ->
        this.addTokenMetrics(token = goldSentence.tokens[i], tokenIndex = i, tree = tree, goldTree = goldTree)
      }

      this.updateCorrectSentences()
    }

    return this.buildStats()
  }

  /**
   * Initialize the metrics counters.
   *
   * @param dependencyTrees a list of dependency trees of parsed sentences
   */
  private fun initCounters(dependencyTrees: List<DependencyTree>) {

    this.counter = MetricsCounter()
    this.counterNoPunct = MetricsCounter()

    this.counter.totalSentences = dependencyTrees.size
    this.counterNoPunct.totalSentences = dependencyTrees.size
    this.counter.totalTokens = dependencyTrees.sumBy { it.elements.count() }
  }

  /**
   * Add the statistic metrics of a given [token].
   *
   * @param token a token of a sentence
   * @param tokenIndex the index of the given [token]
   * @param tree the dependency tree of the parsed sentence
   * @param goldTree the gold dependency tree of the parsed sentence
   */
  private fun addTokenMetrics(token: Token, tokenIndex: Int, tree: DependencyTree, goldTree: DependencyTree) {

    val isNotPunct: Boolean = !this.punctuationRegex.matches(token.word)

    if (isNotPunct) this.counterNoPunct.totalTokens++

    if (tree.heads[tokenIndex] == goldTree.heads[tokenIndex]) {

      this.addCorrectAttachment(isNotPunct)

      if (tree.deprels[tokenIndex] == goldTree.deprels[tokenIndex])
        this.addCorrectLabeledAttachment(isNotPunct)
      else
        this.addUncorrectLabeledAttachment(isNotPunct)

    } else {
      this.addUncorrectAttachment(isNotPunct)
      this.addUncorrectLabeledAttachment(isNotPunct)
    }

    if (tree.posTags[tokenIndex] == goldTree.posTags[tokenIndex]) this.addCorrectPOSTag(isNotPunct)

    if (tree.deprels[tokenIndex]?.softEquals(goldTree.deprels[tokenIndex]) ?:
      (tree.deprels[tokenIndex] == goldTree.deprels[tokenIndex])) this.addCorrectDeprel(isNotPunct)
  }

  /**
   * Add a correct attachment to the current statistic metrics.
   *
   * @param isNotPunct a Boolean indicating if the attachment is related to a non-punctuation token
   */
  private fun addCorrectAttachment(isNotPunct: Boolean) {

    this.counter.unlabeledAttachments++

    if (isNotPunct) this.counterNoPunct.unlabeledAttachments++
  }

  /**
   * Add an uncorrect attachment to the current statistic metrics.
   *
   * @param isNotPunct a Boolean indicating if the attachment is related to a non-punctuation token
   */
  private fun addUncorrectAttachment(isNotPunct: Boolean) {

    this.sentenceMetrics.correctUnlabeled = false

    if (isNotPunct) this.sentenceMetrics.correctUnlabeledNoPunct = false
  }

  /**
   * Add a correct labeled attachment to the current statistic metrics.
   *
   * @param isNotPunct a Boolean indicating if the attachment is related to a non-punctuation token
   */
  private fun addCorrectLabeledAttachment(isNotPunct: Boolean) {

    this.counter.labeledAttachments++

    if (isNotPunct) this.counterNoPunct.labeledAttachments++
  }

  /**
   * Add an uncorrect labeled attachment to the current statistic metrics.
   *
   * @param isNotPunct a Boolean indicating if the attachment is related to a non-punctuation token
   */
  private fun addUncorrectLabeledAttachment(isNotPunct: Boolean) {

    this.sentenceMetrics.correctLabeled = false

    if (isNotPunct) this.sentenceMetrics.correctLabeledNoPunct = false
  }

  /**
   * Add a correct POS tag to the current statistic metrics.
   *
   * @param isNotPunct a Boolean indicating if the POS tag is related to a non-punctuation token
   */
  private fun addCorrectPOSTag(isNotPunct: Boolean) {

    this.counter.correctPOSTags++

    if (isNotPunct) this.counterNoPunct.correctPOSTags++
  }

  /**
   * Add a correct deprel to the current statistic metrics.
   *
   * @param isNotPunct a Boolean indicating if the deprel is related to a non-punctuation token
   */
  private fun addCorrectDeprel(isNotPunct: Boolean) {

    this.counter.correctDeprels++

    if (isNotPunct) this.counterNoPunct.correctDeprels++
  }

  /**
   * Update the counters of correct sentences with the current [sentenceMetrics].
   */
  private fun updateCorrectSentences() {

    if (this.sentenceMetrics.correctLabeled) this.counter.correctLabeledSentences++
    if (this.sentenceMetrics.correctUnlabeled) this.counter.correctUnlabeledSentences++
    if (this.sentenceMetrics.correctLabeledNoPunct) this.counterNoPunct.correctLabeledSentences++
    if (this.sentenceMetrics.correctUnlabeledNoPunct) this.counterNoPunct.correctUnlabeledSentences++
  }

  /**
   * Build the statistics related to the current counted metrics.
   */
  private fun buildStats(): Statistics {

    val punctStats = this.metricsToStatistics(this.counter)
    val noPunctStats = this.metricsToStatistics(this.counterNoPunct)

    return Statistics(
      las = punctStats.las,
      uas = punctStats.uas,
      ps = punctStats.ps,
      ds = punctStats.ds,
      slas = punctStats.slas,
      suas = punctStats.suas,
      noPunctuation = noPunctStats)
  }

  /**
   * @param metricsCounter a metrics counter
   *
   * @return the base statistics object related to the given [metricsCounter]
   */
  private fun metricsToStatistics(metricsCounter: MetricsCounter) = BaseStatistics(
    las = StatMetric(count = metricsCounter.labeledAttachments, total = metricsCounter.totalTokens),
    uas = StatMetric(count = metricsCounter.unlabeledAttachments, total = metricsCounter.totalTokens),
    ps = StatMetric(count = metricsCounter.correctPOSTags, total = metricsCounter.totalTokens),
    ds = StatMetric(count = metricsCounter.correctDeprels, total = metricsCounter.totalTokens),
    slas = StatMetric(count = metricsCounter.correctLabeledSentences, total = metricsCounter.totalSentences),
    suas = StatMetric(count = metricsCounter.correctUnlabeledSentences, total = metricsCounter.totalSentences)
  )

  /**
   * Get the output of the official CoNLL evaluation script, run comparing the sentences represented by the given list
   * of [dependencyTrees] and the [goldSentences].
   *
   * @param dependencyTrees a list of dependency trees of parsed sentences, parallel to the gold sentences
   *
   * @return the output of the official CoNLL evaluation script about the evaluation of the given [dependencyTrees]
   */
  private fun evaluateWithCoNLLScript(dependencyTrees: List<DependencyTree>): String {

    val outputPath: String = this.outputFilePath ?: this.getDefaultOutputPath()
    val conllSentences: List<com.kotlinnlp.conllio.Sentence> = this.goldSentences.zip(dependencyTrees).map {
      (sentence, tree) -> sentence.toCoNLL(dependencyTree = tree, replacePOS = false)
    }

    CoNLLWriter.toFile(sentences = conllSentences, writeComments = false, outputFilePath = outputPath)

    return this.conllEvaluator.evaluate(systemFilePath = outputPath, goldFilePath = this.goldFilePath)!!
  }
}
