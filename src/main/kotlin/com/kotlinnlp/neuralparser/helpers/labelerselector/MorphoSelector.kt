/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.helpers.labelerselector

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Unknown
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar
import com.kotlinnlp.utils.notEmptyOr

/**
 * The selector to use when the labeler predictions are defined as combinations of POS and syntactic types in the
 * Base format.
 */
object MorphoSelector : LabelerSelector {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * Get the list of scored grammatical configurations that are valid for a given attachment.
   *
   * @param configurations the list of grammatical configurations, sorted by descending score
   * @param sentence the input sentence
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   *
   * @return the valid grammatical configurations for the given attachment
   */
  override fun getValidConfigurations(configurations: List<ScoredGrammar>,
                                      sentence: ParsingSentence,
                                      tokenIndex: Int,
                                      headIndex: Int?): List<ScoredGrammar> {

    val possibleMorphologies: Morphologies = sentence.morphoAnalysis!!.allMorphologies[tokenIndex]
    val correctDirection = SyntacticDependency.Direction(tokenIndex = tokenIndex, headIndex = headIndex)
    val possibleConfigurations: List<ScoredGrammar> = configurations.filter { it.config.direction == correctDirection }
    val worstScore: Double = configurations.last().score

    return if (possibleMorphologies.isNotEmpty())
      possibleConfigurations
        .filter { sentence.areConfigurationCompatible(c = it.config, tokenIndex = tokenIndex) }
        .notEmptyOr {
          listOf(ScoredGrammar(
            config = possibleMorphologies.first().buildUnknownConfig(correctDirection),
            score = worstScore))
        }
    else
      possibleConfigurations.filter { it.config.isSingleContentWord() }.notEmptyOr {
        listOf(ScoredGrammar(
          config = GrammaticalConfiguration(GrammaticalConfiguration.Component(
            syntacticDependency = Unknown(correctDirection),
            pos = POSTag.Base(POS.Noun))),
          score = worstScore))
      }
  }

  /**
   * Get the morphologies of a given token that are compatible with the given grammatical configuration.
   *
   * @param sentence the input sentence
   * @param tokenIndex the index of a token of the sentence
   * @param configuration the grammatical configuration of the token
   *
   * @return the morphologies compatible with the given deprel
   */
  override fun getValidMorphologies(sentence: ParsingSentence,
                                    tokenIndex: Int,
                                    configuration: GrammaticalConfiguration): Morphologies {

    val possibleMorphologies = sentence.getCompatibleMorphologies(c = configuration, tokenIndex = tokenIndex)

    return when {

      possibleMorphologies.isNotEmpty() -> possibleMorphologies

      configuration.type == GrammaticalConfiguration.Type.Single -> {

        val pos: POSTag.Base = checkNotNull(configuration.components.single().pos as? POSTag.Base) {
          "The POS must be not null."
        }

        require(pos.type.isContentWord) {
          "Grammatical configuration for tokens without morphological analysis must define a content word."
        }

        Morphologies(Morphology(SingleMorphology(
          lemma = sentence.tokens[tokenIndex].form,
          pos = pos.type,
          allowIncompleteProperties = true)))
      }

      else -> Morphologies()
    }
  }
}
