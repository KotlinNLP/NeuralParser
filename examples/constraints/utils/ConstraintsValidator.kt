/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints.utils

import com.kotlinnlp.conllio.Sentence as CoNLLSentence
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.POS
import com.kotlinnlp.linguisticdescription.morphology.ScoredMorphology
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoSynToken
import com.kotlinnlp.neuralparser.constraints.Constraint
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.helpers.MorphoSynBuilder
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.morphopercolator.MorphoPercolator
import com.kotlinnlp.utils.notEmptyOr
import com.kotlinnlp.utils.progressindicator.ProgressIndicatorBar

/**
 * A helper that verifies constraints on sentences.
 *
 * @param constraints the list of constraints to verify
 * @param sentences the list of sentence to validate
 * @param morphoPreprocessor a morphological sentence preprocessor
 * @param morphoPercolator a percolator of morphology properties
 */
internal class ConstraintsValidator(
  private val constraints: List<Constraint>,
  private val sentences: List<CoNLLSentence>,
  private val morphoPreprocessor: MorphoPreprocessor,
  private val morphoPercolator: MorphoPercolator
) {

  /**
   * Validation info returned from the validator.
   *
   * @property violations the violated constraint associated to the related info
   * @property correctSentences the number of totally correct sentence
   * @property totalSentences the total number of validated sentences
   */
  data class ValidationInfo(
    val violations: Map<Constraint, ViolationInfo>,
    val correctSentences: Int,
    val totalSentences: Int
  ) {

    /**
     * The percentage of correct sentences.
     */
    val correctPerc: Double = 100.0 * this.correctSentences / this.totalSentences
  }

  /**
   * Information about a constraint that has been violated.
   *
   * @property violations the list of violation tokens info
   * @property violatedSentences the count of sentences in which the constraint has been violated
   */
  data class ViolationInfo(
    val violations: MutableList<TokenInfo> = mutableListOf(),
    var violatedSentences: Int = 0)

  /**
   * Information about the token that caused a violation.
   *
   * @property token the morpho-syntactic token of the violation
   * @property conllTokenId the id of the CoNLL token that generated the morpho-syntactic token
   * @property conllSentence the related CoNLL sentence
   */
  data class TokenInfo(
    val token: MorphoSynToken.Single,
    val conllTokenId: Int,
    val conllSentence: CoNLLSentence)

  /**
   * The map of violated constraint associated to the related info.
   */
  private val violationsByConstraint = mutableMapOf<Constraint, ViolationInfo>()

  /**
   * The count of correct sentences.
   */
  private var correct = 0

  /**
   * Verify the [constraints] on the [sentences].
   *
   * @return the validation info
   */
  fun validate(): ValidationInfo {

    val progress = ProgressIndicatorBar(this.sentences.size)

    this.sentences.forEach {
      progress.tick()
      this.processSentence(it)
    }

    return ValidationInfo(
      violations = this.violationsByConstraint,
      correctSentences = this.correct,
      totalSentences = this.sentences.size)
  }

  /**
   * @param sentence a CoNLL sentence to process
   */
  private fun processSentence(sentence: CoNLLSentence) {

    val tokensToCoNLLId: Map<MorphoSynToken.Single, Int> = this.buildMorphoSynTokens(sentence)
    val tokens: List<MorphoSynToken.Single> = tokensToCoNLLId.keys.toList()
    val validator = SentenceValidator(
      constraints = this.constraints,
      tokens = tokens,
      morphoPercolator = morphoPercolator)

    if (violated.isEmpty()) this.correct++

    violated.values.flatMap { it }.toSet().forEach {
      this.violationsByConstraint.getOrPut(it) { ViolationInfo() }.violatedSentences++
    }

    violated.forEach { token, constraints ->
      constraints.forEach { constraint ->
        this.violationsByConstraint.getValue(constraint).violations.add(
          TokenInfo(
            token = token,
            conllTokenId = getCoNLLTokenId(token = token, tokens = tokens),
            conllSentence = sentence))
      }
    }
  }

  /**
   * @param sentence a CoNLL sentence
   *
   * @return a list of flat single morpho-syntactic tokens built from the given sentence
   */
  private fun buildMorphoSynTokens(sentence: CoNLLSentence): List<MorphoSynToken.Single> {

    val sentenceTree = DependencyTree(sentence)
    var nextAvailableId: Int = sentence.tokens.last().id + 1

    val parsingSentence: ParsingSentence = this.morphoPreprocessor.convert(BaseSentence.fromCoNLL(sentence, index = 0))
    val converter = MorphoSynBuilder(parsingSentence = parsingSentence, dependencyTree = sentenceTree)

    val morphoSynTokens: List<MorphoSynToken> = parsingSentence.tokens.mapIndexed { i, parsingToken ->

      val morphoSynToken = converter.buildToken(
        tokenId = parsingToken.id,
        morphologies = this.getValidMorphologies(
          configuration = sentenceTree.getConfiguration(parsingToken.id)!!,
          tokenIndex = i,
          parsingSentence = parsingSentence
        ).map { ScoredMorphology(components = it.components, score = 1.0) })

      if (morphoSynToken is MorphoSynToken.Composite) nextAvailableId += morphoSynToken.components.size

      morphoSynToken
    }

    return flatTokens(morphoSynTokens)
  }

  /**
   * Get the valid morphologies of a token respect to a given grammatical configuration.
   *
   * @param configuration a grammatical configuration of a token
   * @param tokenIndex the index of the token within the sentence tokens
   * @param parsingSentence the parsing sentence of the token
   *
   * @return the list of valid morphologies of the token
   */
  private fun getValidMorphologies(configuration: GrammaticalConfiguration,
                                   tokenIndex: Int,
                                   parsingSentence: ParsingSentence): List<Morphology> {

    val token: ParsingToken = parsingSentence.tokens[tokenIndex]
    val tokenMorphologies: List<Morphology> =
      token.morphologies +
        parsingSentence.multiWords.filter { tokenIndex in it.startToken..it.endToken }.flatMap { it.morphologies }

    return tokenMorphologies
      .filter { configuration.isCompatible(it) }
      .notEmptyOr {

        val posTypes: List<POS> = configuration.components.map { (it.pos as POSTag.Base).type }

        if (!posTypes.first().isComposedBy(POS.Noun))
          println("\n[WARNING] No morphology found for token \"${token.form}\" with pos ${posTypes.joinToString("+")}.")

        // A generic morphology is created when the morphological analyzer did not find it
        listOf(Morphology(
          components = posTypes.map { SingleMorphology(lemma = token.form, pos = it, allowIncompleteProperties = true) }
        ))
      }
  }
}
