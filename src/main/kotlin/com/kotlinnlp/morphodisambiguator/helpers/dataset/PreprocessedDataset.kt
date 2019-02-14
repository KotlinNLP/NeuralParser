package com.kotlinnlp.morphodisambiguator.helpers.dataset

import com.kotlinnlp.linguisticdescription.morphology.properties.*
import com.kotlinnlp.linguisticdescription.morphology.properties.Number
import com.kotlinnlp.linguisticdescription.sentence.SentenceIdentificable
import com.kotlinnlp.morphodisambiguator.language.MorphoToken
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

    private fun convertGender (propertyMap: Map<String, String?>): Gender? {
      val gender: String? = propertyMap.get("gender")
      gender?.let{return MorphologyPropertyFactory(propertyName = "gender", valueAnnotation = it) as Gender}
          ?: return null
    }

    private fun convertNumber (propertyMap: Map<String, String?>): Number? {
      val number: String? = propertyMap.get("number")
      number?.let{return MorphologyPropertyFactory(propertyName = "number", valueAnnotation = it) as Number}
          ?: return null
    }

    private fun convertPerson (propertyMap: Map<String, String?>): Person? {
      val person: String? = propertyMap.get("person")
      person?.let{return MorphologyPropertyFactory(propertyName = "person", valueAnnotation = it) as Person}
          ?: return null
    }

    private fun convertTense (propertyMap: Map<String, String?>): Tense? {
      val tense: String? = propertyMap.get("tense")
      tense?.let{return MorphologyPropertyFactory(propertyName = "tense", valueAnnotation = it) as Tense}
          ?: return null
    }

    private fun convertMood (propertyMap: Map<String, String?>): Mood? {
      val mood: String? = propertyMap.get("mood")
      mood?.let{return MorphologyPropertyFactory(propertyName = "mood", valueAnnotation = it) as Mood}
          ?: return null
    }

    private fun convertMorphologies(token: Dataset.Sentence.Token): List<List<MorphologyProperty?>> {

      val morphologies = mutableListOf<List<MorphologyProperty?>>()

      token.properties.forEach {properties ->
        val morphologyList = mutableListOf<MorphologyProperty?>()
        morphologyList.add(convertGender(properties))
        morphologyList.add(convertNumber(properties))
        morphologyList.add(convertPerson(properties))
        morphologyList.add(convertTense(properties))
        morphologyList.add(convertMood(properties))

        morphologies.add(morphologyList)
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