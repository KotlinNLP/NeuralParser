/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.structuraldistaceparser

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.MorphoSynBuilder
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.combine

/**
 * @property model the parser model
 */
class StructuralDistanceParser(override val model: StructuralDistanceParserModel) : NeuralParser<StructuralDistanceParserModel> {

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
   * The structural distance predictor.
   */
  private val distancePredictor = StructuralDistancePredictor(
    model = this.model.distanceModel,
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
    val pairs: List<Pair<Int, Int>> = contextVectors.indices.toList().combine()
    val distances: List<Double> = this.distancePredictor.forward(StructuralDistancePredictor.Input(
      hiddens = contextVectors,
      pairs = pairs
    ))

    /**
     * Scores are mapped by dependents to governors ids (the root is intended to have id = -1).
     */
    val matrix = mutableMapOf<Int, MutableMap<Int, Double>>()

    sentence.tokens.forEach { matrix[it.id] = mutableMapOf() }

    val dependencyTree = DependencyTree(sentence.tokens.size)

    // TODO

    return MorphoSynBuilder(
      parsingSentence = sentence,
      dependencyTree = dependencyTree
    ).buildSentence()
  }
}
