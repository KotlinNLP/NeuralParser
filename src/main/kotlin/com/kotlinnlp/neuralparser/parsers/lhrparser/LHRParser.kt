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
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.contextencoder.ContextEncoderBuilder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder.HeadsEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.headsencoder.HeadsEncoderBuilder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.DeprelLabeler
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodels.labeler.DeprelLabelerBuilder
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.ArcScores.Companion.rootId
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.CyclesFixer
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.tokensencoder.TokensEncoderFactory

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
   * A more convenient access to the embeddings values of the virtual root.
   */
  private val virtualRoot: DenseNDArray get() = this.model.rootEmbedding.array.values

  /**
   * The builder of the tokens encoder.
   */
  private val tokensEncoderBuilder = TokensEncoderFactory(this.model.tokensEncoderModel, trainingMode = false)

  /**
   * The builder of the [ContextEncoder].
   */
  private val contextEncoderBuilder = ContextEncoderBuilder(this.model.contextEncoderModel)

  /**
   * The builder of the [HeadsEncoder].
   */
  private val headsEncoderBuilder = HeadsEncoderBuilder(this.model.headsEncoderModel)

  /**
   * The builder of the labeler.
   */
  private val deprelLabelerBuilder: DeprelLabelerBuilder? = this.model.labelerModel?.let {
    DeprelLabelerBuilder(model = it)
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
  override fun parse(sentence: Sentence): DependencyTree {

    val encoder: LSSEncoder = this.buildEncoder()
    val lss = encoder.encode(sentence.tokens)
    val scores: ArcScores = CosineDecoder().decode(lss)

    return this.buildDependencyTree(lss, scores)
  }

  /**
   * @return a LSSEncoder
   */
  fun buildEncoder() = LSSEncoder(
    tokensEncoder = this.tokensEncoderBuilder.invoke(),
    contextEncoder = this.contextEncoderBuilder.invoke(),
    headsEncoder = this.headsEncoderBuilder.invoke(),
    virtualRoot = this.virtualRoot)

  /**
   * @param lss the latent syntactic structure
   * @param scores the attachment scores
   *
   * @return the annotated dependency tree (without cycles)
   */
  private fun buildDependencyTree(lss: LatentSyntacticStructure, scores: ArcScores): DependencyTree {

    val dependencyTree = DependencyTree(lss.tokens.size)

    dependencyTree.let {
      this.assignHeads(it, scores)
      this.fixCycles(it, scores)
      this.assignLabels(it, lss)
    }

    return dependencyTree
  }

  /**
   * @param dependencyTree the dependency tree to annotate with the heads
   * @param scores the attachment scores
   */
  fun assignHeads(dependencyTree: DependencyTree, scores: ArcScores) {

    val (topId: Int, topScore: Double) = scores.findHighestScoringTop()

    dependencyTree.setAttachmentScore(dependent = topId, score = topScore)

    scores.filterNot { it.key == topId }.forEach { depId, _ ->

      val (govId: Int, score: Double) = scores.findHighestScoringHead(dependentId = depId, except = listOf(rootId))!!

      dependencyTree.setArc(
        dependent = depId,
        governor = govId,
        allowCycle = true,
        score = score)
    }
  }

  /**
   * @param dependencyTree the dependency tree to fix
   * @param scores the attachment scores
   */
  private fun fixCycles(dependencyTree: DependencyTree, scores: ArcScores) {
    CyclesFixer(dependencyTree, scores).fixCycles()
  }

  /**
   * @param dependencyTree the dependency tree to annotate with pos-tags and deprels
   * @param lss the latent syntactic structure
   */
  private fun assignLabels(dependencyTree: DependencyTree, lss: LatentSyntacticStructure) {

    val labeler: DeprelLabeler? = this@LHRParser.deprelLabelerBuilder?.invoke()

    labeler?.let {
      it.forward(DeprelLabeler.Input(lss, dependencyTree)).forEachIndexed { tokenId, prediction ->
        dependencyTree.setDeprel(tokenId, it.getDeprel(prediction.deprels.argMaxIndex()))
      }
    }
  }
}
