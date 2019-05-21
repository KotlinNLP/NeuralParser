/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.distance

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.helpers.UnlabeledMorphoSynBuilder
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.distance.decoder.DependencyDecoder
import com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn.DeepBiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import kotlin.reflect.KClass
import com.kotlinnlp.neuralparser.parsers.distance.helpers.DistancePredictor.Input as SDPInput

/**
 * A neural parser that takes advantage of the prediction of the distance among tokens.
 *
 * @property model the parser model
 */
class DistanceParser<DecoderType : DependencyDecoder>(
  override val model: DistanceParserModel,
  decoderClass: KClass<DecoderType>
) : NeuralParser<DistanceParserModel> {

  /**
   * The dependency decoder.
   */
  private val decoder: DependencyDecoder = decoderClass.constructors.first().call(this.model)

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
   * Parse a sentence, returning its dependency tree.
   *
   * @param sentence a parsing sentence
   *
   * @return the dependency tree predicted for the given [sentence]
   */
  override fun parse(sentence: ParsingSentence): MorphoSynSentence {

    val tokensEncodings: List<DenseNDArray> = this.tokensEncoder.forward(sentence)
    val contextVectors: List<DenseNDArray> = this.contextEncoder.forward(tokensEncodings).map { it.copy() }

    val dependencyTree = DependencyTree(elements = sentence.tokens.map { it.id } )
    val indexedVectors: List<IndexedValue<DenseNDArray>> =
      sentence.tokens.zip(contextVectors).map { IndexedValue(it.first.id, it.second) }

    this.decoder.decode(indexedVectors).forEach { (govId, depId, score) ->
      dependencyTree.setArc(dependent = depId, governor = govId, score = score)
    }

    return UnlabeledMorphoSynBuilder(
      parsingSentence = sentence,
      dependencyTree = dependencyTree
    ).buildSentence()
  }
}
