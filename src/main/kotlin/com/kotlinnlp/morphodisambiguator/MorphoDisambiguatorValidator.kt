package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.morphodisambiguator.language.MorphoToken
import com.kotlinnlp.morphodisambiguator.language.PropertyNames
import com.kotlinnlp.morphodisambiguator.utils.MetricsCounter
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

class MorphoDisambiguatorValidator (
    val morphoDisambiguator: MorphoDisambiguator,
    val inputSentences: List<ParsingSentence>,
    val goldSentences: List<PreprocessedDataset.MorphoSentence>,
    private val verbose: Boolean = true
){

  /**
   * A counter of statistic metrics.
   */
  private lateinit var counter: MetricsCounter


  /**
   * Get statistics about the evaluation of the disambiguator accuracy on the given [sentences].
   *
   * @return the statistics of the parsing accuracy
   */
  fun evaluate(): MetricsCounter {

    val disambiguatedSentences: List<List<MorphoToken>> = this.disambiguateSentences()

    this.initCounters(this.inputSentences)

    this.goldSentences.zip(disambiguatedSentences).forEach { (goldSentence, disambiguatedSentence) ->

      require(goldSentence.tokens.size == disambiguatedSentence.size) { "The sentences haven't the same size" }

      goldSentence.tokens.zip(disambiguatedSentence).forEach { (goldToken, disambiguatedToken) -> this.addTokenMetrics(
          disambiguatedToken = disambiguatedToken,
          goldToken = goldToken) }

    }

    return this.counter
  }

  /**
   * Disambiguate the validation sentences.
   *
   * @return the list of disambiguated sentences
   */
  private fun disambiguateSentences(): List<List<MorphoToken>> {

    val progress: ProgressIndicatorBar? = if (this.verbose) ProgressIndicatorBar(this.inputSentences.size) else null

    if (this.verbose) println("Start disambiguation of %d sentences:".format(this.inputSentences.size))

    return this.inputSentences.map {sentence ->

      progress?.tick()

      this.morphoDisambiguator.disambiguate(sentence)
    }
  }

  /**
   * Initialize the metrics counters.
   *
   * @param inputSentences a list of disambiguated sentences
   */
  private fun initCounters(inputSentences: List<ParsingSentence>) {

    this.counter = MetricsCounter(corpusMorphologies = this.morphoDisambiguator.model.corpusMorphologies)
    this.counter.totalSentences = inputSentences.size
    this.counter.totalTokens = inputSentences.sumBy { it.tokens.count() }
  }

  /**
   * Add the statistic metrics of a given [token].
   *
   * @param disambiguatedToken a token of a sentence
   * @param goldToken a token of a sentence with correct morphological properties
   */
  private fun addTokenMetrics(disambiguatedToken : MorphoToken, goldToken: MorphoToken) {

    disambiguatedToken.morphoProperties.zip(goldToken.morphoProperties).forEachIndexed {
      i, (propertyPredicted, propertyGold) -> if (propertyGold == propertyPredicted)
      this.addCorrectMorphology(propertyName = PropertyNames().propertyNames[i])
    }
  }

  /**
   * Add a correct attachment to the current statistic metrics.
   *
   * @param propertyName a Boolean indicating if the attachment is related to a non-punctuation token
   */
  private fun addCorrectMorphology(propertyName: String) {

    this.counter.correctMorphologyProperties.put(propertyName,
        this.counter.correctMorphologyProperties.getValue(propertyName) + 1)

  }



}