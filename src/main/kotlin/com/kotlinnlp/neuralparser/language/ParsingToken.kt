/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.language

import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.POSTag
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.ScoredMorphology
import com.kotlinnlp.linguisticdescription.sentence.token.*
import com.kotlinnlp.linguisticdescription.sentence.token.properties.SyntacticRelation
import com.kotlinnlp.linguisticdescription.sentence.token.properties.Position
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector

/**
 * The token of the [ParsingSentence].
 *
 * @property id the id of the token, unique within its sentence
 * @property form the form
 * @property morphologies the list of possible morphologies of the token
 * @property pos the list of part-of-speech tags associated to the token (more for composite tokens, can be null)
 * @param position the position of the token in the text (null if it is a trace)
 */
data class ParsingToken(
  override val id: Int,
  override val form: String,
  override val morphologies: List<Morphology>,
  override val pos: List<POSTag>? = null, // TODO: find a better solution
  private val position: Position?
) : MorphoToken, FormToken, TokenIdentificable {

  /**
   * @param governorId the governor id
   * @param attachmentScore the attachment score
   * @param grammaticalConfiguration the grammatical configuration of this token
   * @param morphoDeprelSelector the morphology and deprel selector
   *
   * @return a new morpho syntactic token
   */
  fun toMutableMorphoSyntacticToken(sentence: ParsingSentence,
                                    tokenIndex: Int,
                                    governorId: Int?,
                                    attachmentScore: Double,
                                    grammaticalConfiguration: GrammaticalConfiguration,
                                    morphoDeprelSelector: MorphoDeprelSelector): MorphoSynToken {

    // TODO: set the score adding the labeler prediction scores of configurations with the same pos
    val morphologies: List<ScoredMorphology> = morphoDeprelSelector.getValidMorphologies(
      sentence = sentence,
      tokenIndex = tokenIndex,
      grammaticalConfiguration = grammaticalConfiguration
    ).map { ScoredMorphology(list = it.components, score = 0.0) }

    require(morphologies.all { it.components.size == grammaticalConfiguration.components.size }) {
      "The selected morphologies must have the same number of components of the given grammatical configuration."
    }

    return if (grammaticalConfiguration.components.size == 1)
      this.buildToken(
        governorId = governorId,
        attachmentScore = attachmentScore,
        grammaticalComponent = grammaticalConfiguration.components.single(),
        morphologies = morphologies)
    else
      MorphoSynToken.Composite(
        id = this.id,
        form = this.form,
        position = checkNotNull(this.position) { "Composite words must have a position." },
        components = grammaticalConfiguration.components.mapIndexed { i, component ->
          this.buildToken(
            governorId = if (i == 0) governorId else null,
            attachmentScore = attachmentScore,
            grammaticalComponent = component,
            morphologies = morphologies.map { it.copy(list = it.components.subList(i, i + 1)) }) as Word
        }
      )
  }

  /**
   *
   */
  private fun buildToken(governorId: Int?,
                         attachmentScore: Double,
                         grammaticalComponent: GrammaticalConfiguration.Component,
                         morphologies: List<ScoredMorphology>): MorphoSynToken {

    val syntacticRelation = SyntacticRelation(
      governor = governorId,
      attachmentScore = attachmentScore,
      dependency = grammaticalComponent.syntacticDependency)

    return if (this.position != null)
      Word(
        id = this.id,
        form = this.form,
        position = this.position,
        pos = grammaticalComponent.pos,
        morphologies = morphologies,
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null) // TODO: set it
    else
      WordTrace(
        id = this.id,
        form = this.form,
        pos = grammaticalComponent.pos,
        morphologies = morphologies,
        syntacticRelation = syntacticRelation,
        coReferences = null, // TODO: set it
        semanticRelations = null)
  }
}
