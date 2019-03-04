package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.morphodisambiguator.helpers.dataset.PreprocessedDataset
import com.kotlinnlp.morphodisambiguator.language.MorphoToken
import com.kotlinnlp.morphodisambiguator.language.PropertyNames
import com.kotlinnlp.morphodisambiguator.utils.Statistics
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

class MorphoDisambiguatorValidator (
    val morphoDisambiguator: MorphoDisambiguator,
    val inputSentences: List<ParsingSentence>,
    val goldSentences: List<PreprocessedDataset.MorphoSentence>,
    private val verbose: Boolean = true
){

  /**
   * A statistics of statistic metrics.
   */
  private lateinit var statistics: Statistics


  /**
   * Get statistics about the evaluation of the disambiguator accuracy on the given [sentences].
   *
   * @return the statistics of the parsing accuracy
   */
  fun evaluate(): Statistics {

    val disambiguatedSentences: List<List<MorphoToken>> = this.disambiguateSentences()

    this.initCounters(this.goldSentences)

    this.goldSentences.zip(disambiguatedSentences).forEach { (goldSentence, disambiguatedSentence) ->

      require(goldSentence.tokens.size == disambiguatedSentence.size) { "The sentences haven't the same size" }

      goldSentence.tokens.zip(disambiguatedSentence).forEach { (goldToken, disambiguatedToken) -> this.addTokenMetrics(
          disambiguatedToken = disambiguatedToken,
          goldToken = goldToken) }

    }

    return this.statistics
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
   * @param goldSentences a list of disambiguated sentences
   */
  private fun initCounters(goldSentences: List<PreprocessedDataset.MorphoSentence>) {

    this.statistics = Statistics(corpusMorphologies = this.morphoDisambiguator.model.corpusMorphologies,
        goldSentences = goldSentences)
    this.statistics.totalSentences = goldSentences.size

  }

  /**
   * Update the statistic metrics of a given [token].
   *
   * @param disambiguatedToken a token of a sentence
   * @param goldToken a token of a sentence with correct morphological properties
   */
  private fun addTokenMetrics(disambiguatedToken : MorphoToken, goldToken: MorphoToken) {

    disambiguatedToken.morphoProperties.zip(goldToken.morphoProperties).forEachIndexed {
      i, (predictedAnnotation, goldAnnotation) ->

      this.addCorrectMorphology(propertyName = PropertyNames().propertyNames[i],
          predictedAnnotation = predictedAnnotation, goldAnnotation = goldAnnotation)
    }
  }

  /**
   * Add a correct prediction to the current statistic metrics.
   *
   * @param propertyName the name of the property
   * @param predictedAnnotation the annotation predictem by the classifier, for the given property
   * @param goldAnnotation the correct (gold) annotation for the given property
   */
  private fun addCorrectMorphology(propertyName: String, predictedAnnotation: String?, goldAnnotation: String?) {

    val metricCounter = this.statistics.metricCounters.get(propertyName)

    if (goldAnnotation != PropertyNames().unknownAnnotation){ // todo may have unknown label

      if (goldAnnotation == predictedAnnotation) {

        if (goldAnnotation != null)
          metricCounter!!.truePos++

      } else {

          if (predictedAnnotation == null)
            metricCounter!!.falseNeg++
          else
            metricCounter!!.falsePos++

      }
    }

  }

}