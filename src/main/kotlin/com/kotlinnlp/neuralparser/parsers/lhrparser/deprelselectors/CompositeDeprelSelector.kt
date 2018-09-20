/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.*
import com.kotlinnlp.linguisticdescription.syntax.SyntacticDependency
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Top
import com.kotlinnlp.linguisticdescription.syntax.dependencies.Unknown
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar
import com.kotlinnlp.neuralparser.utils.notEmptyOr

/**
 * The deprel selector to use with the "composite" deprel format.
 */
class CompositeDeprelSelector : MorphoDeprelSelector {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Get the direction of a syntactic dependency.
     *
     * @param tokenIndex the index of the token to which the deprel must be assigned
     * @param headIndex the index of the token head (can be null)
     *
     * @return the direction of the syntactic dependency between the given token and its head
     */
    private fun getDependencyDirection(tokenIndex: Int, headIndex: Int?): SyntacticDependency.Direction = when {
      headIndex == null -> SyntacticDependency.Direction.ROOT
      tokenIndex < headIndex -> SyntacticDependency.Direction.LEFT
      else -> SyntacticDependency.Direction.RIGHT
    }
  }

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

    // The token morphologies obtained with a morphological analysis.
    val possibleMorphologies: List<Morphology> =
      sentence.tokens[tokenIndex].morphologies + sentence.getTokenMultiWordsMorphologies(tokenIndex)

    val correctDirection: SyntacticDependency.Direction =
      getDependencyDirection(tokenIndex = tokenIndex, headIndex = headIndex)

    // The grammatical configurations compatible with the current dependency direction.
    val possibleConfigurations: List<ScoredGrammar> = configurations.filter { it.config.direction == correctDirection }
    val worstScore: Double = configurations.last().score

    return if (possibleMorphologies.isNotEmpty())
      possibleConfigurations.filter { it.config.isCompatible(possibleMorphologies) }.notEmptyOr {
        listOf(
          possibleMorphologies.first()
            .buildUnknownConfig(tokenIndex = tokenIndex, headIndex = headIndex, score = worstScore))
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
   * Get the morphologies that are compatible with the deprel of a given token.
   *
   * @param token a parsing token
   * @param grammaticalConfiguration the grammatical configuration of the token
   *
   * @return the morphologies compatible with the given deprel
   */
  override fun getValidMorphologies(token: ParsingToken,
                                    grammaticalConfiguration: GrammaticalConfiguration): List<Morphology> {

    val posList: List<POSTag.Base> =
      grammaticalConfiguration.components.filter { it.pos != null }.map { it.pos as POSTag.Base }

    val possibleMorphologies: List<Morphology> = token.morphologies.filter {
      it.components.size == posList.size
        && it.components.zip(posList).all { it.first.pos.baseAnnotation == it.second.type.baseAnnotation }
    }

    return when {

      possibleMorphologies.isNotEmpty() -> possibleMorphologies

      grammaticalConfiguration.type == GrammaticalConfiguration.Type.Single -> {

        require(posList.single().type.isContentWord) {
          "Grammatical configuration for tokens without morphological analysis must define a content word."
        }

        listOf(Morphology(SingleMorphology(
          lemma = token.form,
          pos = posList.single().type,
          allowIncompleteProperties = true)))
      }

      else -> listOf()
    }
  }

  /**
   * Return the morphologies associated to the multi-words that involve a given token.
   *
   * @param tokenIndex the index of the token
   *
   * @return a list of morphologies
   */
  private fun ParsingSentence.getTokenMultiWordsMorphologies(tokenIndex: Int): List<Morphology> =
    this.multiWords
      .filter { tokenIndex in it.startToken..it.endToken }
      .flatMap { it.morphologies }

  /**
   * @param possibleMorphologies the list of possible morphologies of the token to which this deprel is assigned
   *
   * @return whether this deprel is compatible with the given morphologies
   */
  private fun GrammaticalConfiguration.isCompatible(possibleMorphologies: List<Morphology>): Boolean =
    possibleMorphologies.any {
      it.components.size == this.components.size &&
        it.components.zip(this.components).all {
          it.first.pos.baseAnnotation == (it.second.pos as POSTag.Base).type.baseAnnotation
        }
    }

  /**
   * Build a grammatical configuration from this [Morphology] with TOP or UNKNOWN dependency.
   *
   * @param tokenIndex the index of the token to which the deprel must be assigned
   * @param headIndex the index of the token head (can be null)
   * @param score the score to assign to the new deprel
   *
   * @return a new grammatical configuration built from this morphology
   */
  private fun Morphology.buildUnknownConfig(tokenIndex: Int, headIndex: Int?, score: Double): ScoredGrammar {

    val direction: SyntacticDependency.Direction =
      getDependencyDirection(tokenIndex = tokenIndex, headIndex = headIndex)

    return ScoredGrammar(
      config = GrammaticalConfiguration(components = this.components.mapIndexed { i, it ->
        GrammaticalConfiguration.Component(
          syntacticDependency = if (i == 0) Top(direction) else Unknown(direction),
          pos = POSTag.Base(POS.byBaseAnnotation(it.pos.baseAnnotation)))
      }),
      score = score
    )
  }

  /**
   * @return whether this deprel defines a content word with a single morphology
   */
  private fun GrammaticalConfiguration.isSingleContentWord(): Boolean =
    this.type == GrammaticalConfiguration.Type.Single
      && (this.components.single().pos as POSTag.Base).type.isContentWord
}
