/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.constraints.Constraint
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.headsencoder.HeadsEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.DeprelLabeler
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.decoders.CosineDecoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.helpers.DependencyTreeBuilder

/**
 * The Latent Head Representation (LHR) Parser.
 *
 * Implemented as described in the following publication:
 *   [Non-Projective Dependency Parsing via Latent Heads Representation (LHR)](https://arxiv.org/abs/1802.02116)
 *
 * @property model the parser model
 * @param constraints a list of linguistic constraints (can be null)
 */
class LHRParser(override val model: LHRModel, val constraints: List<Constraint>? = null) : NeuralParser<LHRModel> {

  /**
   * The Encoder of the Latent Syntactic Structure.
   */
  private val lssEncoder = LSSEncoder(
    tokensEncoderWrapper = this.model.tokensEncoderWrapperModel.buildWrapper(useDropout = false),
    contextEncoder = ContextEncoder(this.model.contextEncoderModel, useDropout = false),
    headsEncoder = HeadsEncoder(this.model.headsEncoderModel, useDropout = false),
    virtualRoot = this.model.rootEmbedding.array.values)

  /**
   * The builder of the labeler.
   */
  private val deprelLabeler: DeprelLabeler? = this.model.labelerModel?.let {
    DeprelLabeler(it, useDropout = false)
  }

  /**
   * Parse a sentence, returning its dependency tree.
   * The dependency tree is obtained by decoding a latent syntactic structure.
   * If the labeler is available, the dependency tree can contains deprel and posTag annotations.
   *
   * @param sentence a parsing sentence
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: ParsingSentence): MorphoSyntacticSentence {

    val lss: LatentSyntacticStructure = this.lssEncoder.encode(sentence)

    val dependencyTree: DependencyTree = DependencyTreeBuilder(
      lss = lss,
      scoresMap = CosineDecoder().decode(lss),
      deprelLabeler = this.deprelLabeler,
      constraints = this.constraints,
      morphoDeprelSelector = this.model.morphoDeprelSelector,
      deprelScoreThreshold = 1.0 / this.model.corpusDictionary.deprelTags.size
    ).build()

    return sentence.toMorphoSyntacticSentence(
      dependencyTree = dependencyTree,
      morphoDeprelSelector = this.model.morphoDeprelSelector)
  }
}
