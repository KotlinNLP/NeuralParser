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
 * The selector to use when the labeler predictions are defined as combinations of POS and syntactic types in the
 * Base format.
 */
object CompositeSelector : LabelerSelector {

  /**
   * A helper that manages the morphologies of a [ParsingToken] and checks their compatibilities with a
   * [GrammaticalConfiguration].
   *
   * @param sentence a sentence
   * @param tokenIndex the index of a token of the sentence
   */
  private class TokenMorphologies(sentence: ParsingSentence, tokenIndex: Int) {

    /**
     * The focus token.
     */
    val token: ParsingToken = sentence.tokens[tokenIndex]

    /**
     * The union of the [token] morphologies and the morphologies of multi-words that the [token] introduces.
     */
    val startMorphologies: List<Morphology> = this.token.morphologies + sentence.getTokenStartMWMorphologies(tokenIndex)

    /**
     * The morphologies of multi-words that involve the [token], but do not start with.
     */
    val middleMWMorphologies: List<Morphology> = sentence.getTokenMiddleMWMorphologies(tokenIndex)

    /**
     * All the possible morphologies of the token (multi-words included).
     */
    val possibleMorphologies: List<Morphology> = this.startMorphologies + this.middleMWMorphologies

    /**
     * Check whether the morphologies of the [token] are compatible with the given configuration.
     * Middle multi-words morphologies are compared partially (only with the "CONTIN" components.
     *
     * @param configuration a grammatical configuration
     *
     * @return true if the morphologies of the [token] are compatible with the given configuration, otherwise false
     */
    fun areCompatible(configuration: GrammaticalConfiguration): Boolean =
      this.startMorphologies.any { configuration.isCompatible(it) } ||
        this.middleMWMorphologies.any { configuration.isPartiallyCompatible(it) }

    /**
     * @param configuration a grammatical configuration
     *
     * @return the [token] morphologies that are compatible with the given configuration
     */
    fun getCompatible(configuration: GrammaticalConfiguration): List<Morphology> =
      this.possibleMorphologies.filter { configuration.isCompatible(it) }

    /**
     * Return the morphologies of the multi-words that start with a given token.
     *
     * @param tokenIndex the index of the token
     *
     * @return a list of morphologies
     */
    private fun ParsingSentence.getTokenStartMWMorphologies(tokenIndex: Int): List<Morphology> =
      this.multiWords
        .filter { tokenIndex == it.startToken }
        .flatMap { it.morphologies }

    /**
     * Return the morphologies of the multi-words that involve a given token, but do not start with it.
     *
     * @param tokenIndex the index of the token
     *
     * @return a list of morphologies
     */
    private fun ParsingSentence.getTokenMiddleMWMorphologies(tokenIndex: Int): List<Morphology> =
      this.multiWords
        .filter { tokenIndex in (it.startToken + 1)..it.endToken }
        .flatMap { it.morphologies }
  }

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

    val tokenMorphologies = TokenMorphologies(sentence = sentence, tokenIndex = tokenIndex)

    val correctDirection: SyntacticDependency.Direction =
      getDependencyDirection(tokenIndex = tokenIndex, headIndex = headIndex)

    // The grammatical configurations compatible with the current dependency direction.
    val possibleConfigurations: List<ScoredGrammar> = configurations.filter { it.config.direction == correctDirection }
    val worstScore: Double = configurations.last().score

    return if (tokenMorphologies.possibleMorphologies.isNotEmpty())
      possibleConfigurations
        .filter { tokenMorphologies.areCompatible(it.config) }
        .notEmptyOr {
          listOf(
            tokenMorphologies.possibleMorphologies.first()
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
   * Get the morphologies of a given token that are compatible with the given grammatical configurations.
   *
   * @param sentence the input sentence
   * @param tokenIndex the index of a token of the sentence
   * @param grammaticalConfiguration the grammatical configuration of the token
   *
   * @return the morphologies compatible with the given deprel
   */
  override fun getValidMorphologies(sentence: ParsingSentence,
                                    tokenIndex: Int,
                                    grammaticalConfiguration: GrammaticalConfiguration): List<Morphology> {

    val possibleMorphologies: List<Morphology> =
      TokenMorphologies(sentence = sentence, tokenIndex = tokenIndex).getCompatible(grammaticalConfiguration)

    return when {

      possibleMorphologies.isNotEmpty() -> possibleMorphologies

      grammaticalConfiguration.type == GrammaticalConfiguration.Type.Single -> {

        val pos: POSTag.Base = grammaticalConfiguration.components.single().pos as POSTag.Base

        require(pos.type.isContentWord) {
          "Grammatical configuration for tokens without morphological analysis must define a content word."
        }

        listOf(Morphology(SingleMorphology(
          lemma = sentence.tokens[tokenIndex].form,
          pos = pos.type,
          allowIncompleteProperties = true)))
      }

      else -> listOf()
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

  /**
   * @return whether this deprel defines a content word with a single morphology
   */
  private fun GrammaticalConfiguration.isSingleContentWord(): Boolean =
    this.type == GrammaticalConfiguration.Type.Single
      && (this.components.single().pos as POSTag.Base).type.isContentWord
}
