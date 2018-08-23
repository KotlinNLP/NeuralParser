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
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.contextencoder.ContextEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.headsencoder.HeadsEncoder
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.DeprelLabeler
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.ArcScores.Companion.rootId
import com.kotlinnlp.neuralparser.parsers.lhrparser.utils.CyclesFixer
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.deprelselectors.MorphoDeprelSelector
import com.kotlinnlp.neuralparser.traces.CoordCorefHelper
import com.kotlinnlp.neuralparser.traces.RuleBasedTracesHandler
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.cosineSimilarity
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

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
   * The auto-encoder used to score a dependency of a token.
   */
  internal val dependencyAutoEncoder = FeedforwardNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.model.dependencyAutoEncoder,
    useDropout = false,
    propagateToInput = false)

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

    val parsedSentence = sentence.toMorphoSyntacticSentence(
      dependencyTree = this.buildDependencyTree(lss),
      morphoDeprelSelector = this.model.morphoDeprelSelector)

    // TODO: move the trace handler into the model?
    RuleBasedTracesHandler(listOf(CoordCorefHelper())).addTraces(parsedSentence)

    return parsedSentence
  }

  /**
   * Decode the dependency tree of a given latent syntactic structure.
   *
   * @param lss the latent syntactic structure encoded from the input sentence
   *
   * @return the annotated dependency tree of the given LSS
   */
  private fun buildDependencyTree(lss: LatentSyntacticStructure): DependencyTree {

    val dependencyTree = DependencyTree(lss.sentence.tokens.map { it.id })
    val scores: ArcScores = CosineDecoder().decode(lss)

    dependencyTree.apply {
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

    scores.keys.filter { it != topId }.forEach { depId ->

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
  private fun DependencyTree.assignLabels(lss: LatentSyntacticStructure) {

    val selector: MorphoDeprelSelector = this@LHRParser.model.morphoDeprelSelector

    this@LHRParser.deprelLabeler?.let { labeler ->

      val scores: List<Double> = labeler.predict(DeprelLabeler.Input(lss, this)).mapIndexed { tokenIndex, prediction ->

        val tokenId: Int = this.elements[tokenIndex]
        val (deprelIndex, deprel) = selector.getBestDeprel(
          deprels = prediction,
          sentence = lss.sentence,
          tokenIndex = tokenIndex,
          headIndex = this.getHead(tokenId)?.let { this.getPosition(it) })

        this.setDeprel(dependent = tokenId, deprel = deprel)

        this@LHRParser.calcDependencyScore(lss = lss, tokenId = tokenId, bestDeprelIndex = deprelIndex)
      }

      this.score = scores.average()
    }
  }

  /**
   * Calculate the score of the tree dependency of a given token.
   *
   * @param lss the latent syntactic structure of a sentence
   * @param tokenId the id of a token of the sentence
   * @param bestDeprelIndex the index of the best deprel predicted for the given token
   *
   * @return the dependency score of the given token
   */
  private fun calcDependencyScore(lss: LatentSyntacticStructure, tokenId: Int, bestDeprelIndex: Int): Double {

    val numOfDeprels: Int = this.model.corpusDictionary.deprelTags.size

    val features: DenseNDArray = concatVectorsV(
      lss.getContextVectorById(tokenId),
      lss.getLatentHeadById(tokenId),
      if (bestDeprelIndex >= 0)
        DenseNDArrayFactory.oneHotEncoder(length = numOfDeprels, oneAt = bestDeprelIndex)
      else
        DenseNDArrayFactory.zeros(Shape(numOfDeprels))
    )

    return cosineSimilarity(this.dependencyAutoEncoder.forward(features).normalize2(), features.normalize2())
  }
}
