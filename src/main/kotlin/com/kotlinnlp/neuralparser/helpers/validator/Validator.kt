/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.validator

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.conllio.Token as CoNLLToken
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.preprocessors.SentencePreprocessor
import com.kotlinnlp.neuralparser.helpers.statistics.MetricsCounter
import com.kotlinnlp.neuralparser.helpers.statistics.SentenceMetrics
import com.kotlinnlp.neuralparser.helpers.statistics.Statistics
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * The Validator.
 *
 * @param neuralParser the neural parser
 * @property sentences the sentences to parse containing the gold annotation
 * @param sentencePreprocessor the sentence preprocessor (e.g. to perform morphological analysis)
 * @property verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class Validator(
  private val neuralParser: NeuralParser<*>,
  val sentences: List<CoNLLSentence>,
  sentencePreprocessor: SentencePreprocessor,
  private val verbose: Boolean = true
) {

  companion object {

    /**
     * The regular expression to match punctuation forms.
     */
    val punctuationRegex = Regex("^[-!\"#%&'()*,./:;?@\\[\\]_{}]+$")
  }

  init {
    require(sentences.all { it.hasAnnotatedHeads() }) {
      "A gold sentence must have a dependency tree with all heads annotated."
    }
  }

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
   * The parser wrapper to parse sentences in CoNLL format.
   */
  private val conllParser = CoNLLDependencyParser(
    neuralParser = this.neuralParser,
    sentencePreprocessor = sentencePreprocessor)

  /**
   * Get statistics about the evaluation of the parsing accuracy on the given [sentences].
   *
   * @return the statistics of the parsing accuracy
   */
  fun evaluate(): Statistics {

    val parsedSentences: List<CoNLLSentence> = this.parseSentences()

    this.initCounters(parsedSentences)

    this.sentences.zip(parsedSentences).forEach { (goldSentence, parsedSentence) ->

      val goldTree: DependencyTree = this.buildTree(goldSentence)
      val parsedTree: DependencyTree = this.buildTree(parsedSentence, allowCycles = true)

      require(parsedTree.size == goldTree.size) { "The dependency tree and its gold haven't the same size" }

      this.sentenceMetrics = SentenceMetrics()

      goldSentence.tokens.forEach { this.addTokenMetrics(token = it, parsedTree = parsedTree, goldTree = goldTree) }

      this.updateCorrectSentences()
    }

    return this.buildStats()
  }

  /**
   * @param sentence a CoNLL sentence
   * @param allowCycles if true it allows to create cycles when building the tree
   *
   * @return a new dependency tree based on the given sentence
   */
  private fun buildTree(sentence: CoNLLSentence, allowCycles: Boolean = false): DependencyTree =
    if (this.neuralParser.labellingEnabled)
      DependencyTree.Labeled(sentence = sentence, allowCycles = allowCycles)
    else
      DependencyTree.Unlabeled(sentence = sentence, allowCycles = allowCycles)

  /**
   * Parse the validation CoNLL sentences.
   *
   * @return the list of parsed CoNLL sentences
   */
  private fun parseSentences(): List<CoNLLSentence> {

    val progress: ProgressIndicatorBar? = if (this.verbose) ProgressIndicatorBar(this.sentences.size) else null

    if (this.verbose) println("Start parsing of %d sentences:".format(this.sentences.size))

    return this.sentences.mapIndexed { i, sentence ->

      progress?.tick()

      this.conllParser.parse(sentence, index = i)
    }
  }

  /**
   * Initialize the metrics counters.
   *
   * @param parsedSentences a list of parsed sentences
   */
  private fun initCounters(parsedSentences: List<CoNLLSentence>) {

    this.counter = MetricsCounter()
    this.counterNoPunct = MetricsCounter()

    this.counter.totalSentences = parsedSentences.size
    this.counterNoPunct.totalSentences = parsedSentences.size
    this.counter.totalTokens = parsedSentences.sumBy { it.tokens.count() }
  }

  /**
   * Add the statistic metrics of a given [token].
   *
   * @param token a token of a sentence
   * @param parsedTree the dependency tree of the parsed sentence
   * @param goldTree the gold dependency tree of the parsed sentence
   */
  private fun addTokenMetrics(token: CoNLLToken, parsedTree: DependencyTree, goldTree: DependencyTree) {

    val isNotPunct: Boolean = !punctuationRegex.matches(token.form)
    val parsedConfig: GrammaticalConfiguration? = (parsedTree as? DependencyTree.Labeled)?.getConfiguration(token.id)
    val goldConfig: GrammaticalConfiguration? = (goldTree as? DependencyTree.Labeled)?.getConfiguration(token.id)
    val parsedDependencies: List<SyntacticDependency>? =
      parsedConfig?.components?.map { it.syntacticDependency }
    val goldDependencies: List<SyntacticDependency>? =
      (goldTree as? DependencyTree.Labeled)?.getConfiguration(token.id)?.components?.map { it.syntacticDependency }

    if (isNotPunct) this.counterNoPunct.totalTokens++

    if (parsedTree.getHead(token.id) == goldTree.getHead(token.id)) {

      this.addCorrectAttachment(isNotPunct)

      if (parsedDependencies == goldDependencies)
        this.addCorrectLabeledAttachment(isNotPunct)
      else
        this.addUncorrectLabeledAttachment(isNotPunct)

    } else {
      this.addUncorrectAttachment(isNotPunct)
      this.addUncorrectLabeledAttachment(isNotPunct)
    }

    if (parsedConfig?.components?.map { it.pos } == goldConfig?.components?.map { it.pos })
      this.addCorrectPOSTag(isNotPunct)

    if ((parsedDependencies != null && goldDependencies != null
        && parsedDependencies.zip(goldDependencies).all { it.first.softEquals(it.second) })
      || (parsedDependencies == goldDependencies))
      this.addCorrectDeprel(isNotPunct)
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

    val punctStats = this.counter.toStatistics()
    val noPunctStats = this.counterNoPunct.toStatistics()

    return Statistics(
      las = punctStats.las,
      uas = punctStats.uas,
      ps = punctStats.ps,
      ds = punctStats.ds,
      slas = punctStats.slas,
      suas = punctStats.suas,
      noPunctuation = noPunctStats)
  }
}
