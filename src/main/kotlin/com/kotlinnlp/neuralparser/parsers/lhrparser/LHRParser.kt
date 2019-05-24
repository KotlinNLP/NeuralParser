/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.lssencoder.LSSEncoder
import com.kotlinnlp.lssencoder.LatentSyntacticStructure
import com.kotlinnlp.lssencoder.decoder.CosineDecoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.Labeler
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.sentencebuilder.LabeledMorphoSynBuilder
import com.kotlinnlp.neuralparser.helpers.sentencebuilder.UnlabeledMorphoSynBuilder
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.GreedyDependencyTreeBuilder

/**
 * The Latent Head Representation (LHR) Parser.
 *
 * Implemented as described in the following publication:
 *   [Non-Projective Dependency Parsing via Latent Heads Representation (LHR)](https://arxiv.org/abs/1802.02116)
 *
 * @property model the parser model
 */
class LHRParser(override val model: LHRModel) : NeuralParser<LHRModel> {

  /**
   * Whether this parser executes the morpho-syntactic labelling.
   */
  override val labellingEnabled: Boolean = this.model.useLabeler

  /**
   * The Encoder of the Latent Syntactic Structure.
   */
  private val lssEncoder = LSSEncoder(this.model.lssModel, useDropout = false)

  /**
   * The builder of the labeler.
   */
  private val labeler: Labeler? = this.model.labelerModel?.let { Labeler(model = it, useDropout = false) }

  /**
   * Parse a sentence, returning its dependency tree.
   * The dependency tree is obtained by decoding a latent syntactic structure.
   * If the labeler is available, the dependency tree could contain grammatical information.
   *
   * @param sentence a parsing sentence
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: ParsingSentence): MorphoSynSentence {

    val lss: LatentSyntacticStructure<ParsingToken, ParsingSentence> = this.lssEncoder.forward(sentence)

    val dependencyTree: DependencyTree = GreedyDependencyTreeBuilder(
      lss = lss,
      scoresMap = CosineDecoder().decode(lss),
      labeler = this.labeler
    ).build()

    return when (dependencyTree) {

      is DependencyTree.Labeled ->
        LabeledMorphoSynBuilder(parsingSentence = sentence, dependencyTree = dependencyTree).buildSentence()

      is DependencyTree.Unlabeled ->
        UnlabeledMorphoSynBuilder(parsingSentence = sentence, dependencyTree = dependencyTree).buildSentence()
    }
  }
}
