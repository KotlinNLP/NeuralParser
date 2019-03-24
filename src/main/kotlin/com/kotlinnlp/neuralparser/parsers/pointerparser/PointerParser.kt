/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.pointerparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.MorphoSynBuilder
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork.PointerNetworkProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 *
 * @property model the parser model
 */
class PointerParser(override val model: PointerParserModel) : NeuralParser<PointerParserModel> {

  /**
   * The tokens encoder.
   */
  private val tokensEncoder = this.model.tokensEncoderModel.buildEncoder(useDropout = false)

  /**
   * The encoder of tokens encodings sentential context.
   */
  private val contextEncoder = DeepBiRNNEncoder<DenseNDArray>(
    network = this.model.contextEncoderModel,
    propagateToInput = true,
    useDropout = false)

  /**
   * The positional encoder.
   */
  private val depsPointer = PointerNetworkProcessor(this.model.depsPointerNetworkModel)

  /**
   * The positional encoder.
   */
  private val headsPointer = PointerNetworkProcessor(this.model.headsPointerNetworkModel)

  /**
   * The labeler.
   */
  private val labeler = Labeler(
    model = this.model.labelerNetworkModel,
    grammaticalConfigurations = this.model.grammaticalConfigurations,
    useDropout = false)

  /**
   * Parse a sentence, returning its dependency tree.
   *
   * @param sentence a parsing sentence
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: ParsingSentence): MorphoSynSentence {

    val tokensEncodings: List<DenseNDArray> = this.tokensEncoder.forward(sentence)
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings).map { it.copy() }
    val tokensDependents: List<DenseNDArray> = this.depsPointer.forwardNoActivation(contextVectors)
    val tokensHeads: List<DenseNDArray> = this.headsPointer.forwardNoActivation(contextVectors)

    /**
     * Scores are mapped by dependents to governors ids (the root is intended to have id = -1).
     */
    val matrix = mutableMapOf<Int, MutableMap<Int, Double>>()

    sentence.tokens.forEach { matrix[it.id] = mutableMapOf() }

    tokensDependents.forEachIndexed { tokenIndex, depsScores ->
      depsScores.toDoubleArray().forEachIndexed { depIndex, depScore ->
        if (depIndex != tokenIndex) {
          matrix.getValue(sentence.tokens[depIndex].id)[sentence.tokens[tokenIndex].id] = depScore
        }
      }
    }

    tokensHeads.forEachIndexed { tokenIndex, headsScores ->
      headsScores.toDoubleArray().forEachIndexed { headIndex, headScore ->

        if (headIndex != tokenIndex) {
          val a = matrix.getValue(sentence.tokens[tokenIndex].id)[sentence.tokens[headIndex].id]!!
          val c = (a + headScore) / 2.0
          matrix.getValue(sentence.tokens[tokenIndex].id)[sentence.tokens[headIndex].id] = c
        }

      }
    }

    val dependencyTree = GreedyDependencyTreeBuilder(sentence, ScoredArcs(matrix)).build()

    this.assignLabels(sentence, contextVectors, dependencyTree)

    return MorphoSynBuilder(
      parsingSentence = sentence,
      dependencyTree = dependencyTree
    ).buildSentence()
  }

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  private fun PointerNetworkProcessor.forwardNoActivation(input: List<DenseNDArray>): List<DenseNDArray> {

    this.setInputSequence(input)

    return input.map {
      this.forward(it)
      this.getLastNotActivatedImportanceScores()
    }
  }

  /**
   * Labels the dependency tree
   */
  private fun assignLabels(sentence: ParsingSentence,
                           contextVectors: List<DenseNDArray>,
                           dependencyTree: DependencyTree) {

    this.labeler.predict(Labeler.Input(sentence, contextVectors, dependencyTree)).forEach { tokenId, configurations ->
      dependencyTree.setGrammaticalConfiguration(
        dependent = tokenId,
        configuration = configurations.first().config
      )
    }
  }
}
