/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser

import com.kotlinnlp.neuralparser.parsers.lhrparser.decoders.CosineDecoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.ArcScores
import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSyntacticSentence
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder.HeadsEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.DeprelLabeler
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.ArcScores.Companion.rootId
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.CyclesFixer
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.ParsingSentence

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
   * @param sentence a [Sentence]
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: ParsingSentence): MorphoSyntacticSentence {

    val lss: LatentSyntacticStructure = this.lssEncoder.encode(sentence)
    val scores: ArcScores = CosineDecoder().decode(lss)
    val dependencyTree: DependencyTree = this.buildDependencyTree(lss, scores)

    return sentence.toMorphoSyntacticSentence(dependencyTree)
  }

  /**
   * @param lss the latent syntactic structure
   * @param scores the attachment scores
   *
   * @return the annotated dependency tree (without cycles)
   */
  private fun buildDependencyTree(lss: LatentSyntacticStructure, scores: ArcScores): DependencyTree {

    val dependencyTree = DependencyTree(lss.size)

    with(dependencyTree) {
      assignHeads(scores)
      fixCycles(scores)
      assignLabels(lss)
    }

    return dependencyTree
  }

  /**
   * Assign the heads to this dependency tree using the highest scoring arcs from the given [scores].
   *
   * @param scores the attachment scores
   */
  private fun DependencyTree.assignHeads(scores: ArcScores) {

    val (topId: Int, topScore: Double) = scores.findHighestScoringTop()

    this.setAttachmentScore(dependent = topId, score = topScore)

    scores.filterNot { it.key == topId }.forEach { depId, _ ->

      val (govId: Int, score: Double) = scores.findHighestScoringHead(dependentId = depId, except = listOf(rootId))!!

      this.setArc(
        dependent = depId,
        governor = govId,
        allowCycle = true,
        score = score)
    }
  }

  /**
   * Fix possible cycles using the given [scores].
   */
  private fun DependencyTree.fixCycles(scores: ArcScores) = CyclesFixer(this, scores).fixCycles()

  /**
   * Annotate this dependency tree with the labels.
   *
   * @param lss the latent syntactic structure
   */
  private fun DependencyTree.assignLabels(lss: LatentSyntacticStructure) = this@LHRParser.deprelLabeler?.let { labeler ->
    labeler.predict(DeprelLabeler.Input(lss, this)).forEachIndexed { tokenId, prediction ->
      this.setDeprel(tokenId, prediction.first().value)
    }
  }
}
