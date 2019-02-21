package com.kotlinnlp.morphodisambiguator.helpers.dataset

import com.kotlinnlp.linguisticdescription.morphology.properties.*
import com.kotlinnlp.linguisticdescription.morphology.properties.Number
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.morphodisambiguator.language.MorphoToken
import com.kotlinnlp.morphodisambiguator.language.PropertyNames
import com.kotlinnlp.neuralparser.helpers.labelerselector.LabelerSelector
import com.kotlinnlp.neuralparser.helpers.labelerselector.NoFilterSelector
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken

class PreprocessedDataset (
    val morphoExamples: List<MorphoSentence>,
    val parsingExamples: List<ParsingSentence>
){

  data class MorphoSentence(override val tokens: List<MorphoToken>): SentenceIdentificable<MorphoToken>()

  companion object {

    private fun convertMorphologies(token: Dataset.Sentence.Token): List<String?> {

      val morphologies = mutableListOf<String?>()
      val property = token.properties.last()

      PropertyNames().propertyNames.forEach{
        morphologies.add(property.get(it))
      }

      return morphologies
    }

    private fun buildMorphoSentenceFromJson(sentence: Dataset.Sentence): MorphoSentence{
      val preprocessedSentence = mutableListOf<MorphoToken>()
      var i = 0

      sentence.tokens?.forEach {token ->

        preprocessedSentence.add(MorphoToken(
            id = i,
            form = token.form,
            morphoProperties = convertMorphologies(token)))
        i += 1
      }

      return MorphoSentence(tokens = preprocessedSentence)

    }

    private fun buildParsingSentenceFromJson(sentence: Dataset.Sentence): ParsingSentence{
      val preprocessedSentence = mutableListOf<ParsingToken>()
      var i = 0

      sentence.tokens?.forEach {token ->

        preprocessedSentence.add(ParsingToken(
            id = i,
            form = token.form,
            position = null))
        i += 1
      }

      return ParsingSentence(tokens = preprocessedSentence, labelerSelector = NoFilterSelector)

    }

    /**
     * Build a [PreprocessedDataset] from a [Dataset], encoding the morphological properties for each token.
     *
     * @param dataset a dataset
     *
     * @return an encoded dataset
     */
    fun fromDataset(dataset: Dataset): PreprocessedDataset {

      val mSentences = mutableListOf<MorphoSentence>()
      val pSentences = mutableListOf<ParsingSentence>()

      dataset.examples.forEach {sentence -> mSentences.add(buildMorphoSentenceFromJson(sentence))}
      dataset.examples.forEach {sentence -> pSentences.add(buildParsingSentenceFromJson(sentence))}

      return PreprocessedDataset(morphoExamples = mSentences, parsingExamples = pSentences)
    }
  }
}