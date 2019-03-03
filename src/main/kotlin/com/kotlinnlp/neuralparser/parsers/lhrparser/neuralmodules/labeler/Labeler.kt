/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.lssencoder.LatentSyntacticStructure
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.parsers.lhrparser.neuralmodules.labeler.utils.ScoredGrammar
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.utils.notEmptyOr

/**
 * The Labeler.
 *
 * @param model the model of this labeler
 * @property useDropout whether to apply the dropout during the forward
 * @property id an identification number useful to track a specific encoder
 */
class Labeler(
  private val model: LabelerModel,
  override val useDropout: Boolean,
  override val id: Int = 0
) : NeuralProcessor<
  Labeler.Input, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  Labeler.InputErrors, // InputErrorsType
  LabelerParams> { // ParamsType

  /**
   * The input of this labeler.
   *
   * @param lss the latent syntactic structure
   * @param dependencyTree the dependency tree
   */
  class Input(val lss: LatentSyntacticStructure<*, *>, val dependencyTree: DependencyTree)

  /**
   * This encoder propagate the errors to the input.
   */
  override val propagateToInput: Boolean = true

  /**
   * The input errors of this labeler.
   *
   * @param rootErrors the errors of the virtual root
   * @param contextErrors the errors of the context vectors
   */
  class InputErrors(
    val rootErrors: DenseNDArray,
    val contextErrors: List<DenseNDArray>)

  /**
   * The processor that classify the grammar of a token.
   */
  private val processor = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.model.networkModel,
    useDropout = this.useDropout,
    propagateToInput = true)

  /**
   * The dependency tree of the last input, used during the training.
   */
  private lateinit var dependencyTree: DependencyTree

  /**
   * Score the possible grammatical configurations of each token of a given input.
   *
   * @param input a [Labeler] input
   *
   * @return a map of valid grammatical configurations (sorted by descending score) associated to each token id
   */
  fun predict(input: Input): Map<Int, List<ScoredGrammar>> {

    return this.forward(input)
      .asSequence()
      .map { it.toScoredGrammar() }
      .withIndex()
      .associate { (tokenIndex, configurations) ->

        val tokenId: Int = input.dependencyTree.elements[tokenIndex]

        val validConfigurations: List<ScoredGrammar> = (input.lss.sentence as ParsingSentence).getValidConfigurations(
          tokenIndex = tokenIndex,
          headIndex = input.dependencyTree.getHead(tokenId)?.let { input.dependencyTree.getPosition(it) },
          configurations = configurations)

        tokenId to validConfigurations
          .filter { it.score >= this.model.labelerScoreThreshold }
          .notEmptyOr { validConfigurations.subList(0, 1) }
      }
  }


  /**
   * Return the network outcomes for each token.
   *
   * @param input a [Labeler] input
   *
   * @return the network outcomes for each token
   */
  override fun forward(input: Input): List<DenseNDArray> {

    this.dependencyTree = input.dependencyTree

    return this.processor.forward(featuresListBatch = ArrayList(input.lss.sentence.tokens.map {
      this.extractFeatures(tokenId = it.id, lss = input.lss)
    }))
  }

  /**
   * Propagate the errors through the neural components of the labeler.
   *
   * @param outputErrors the list of errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    this.processor.backward(outputErrors)
  }

  /**
   * @return the input errors and the root errors
   */
  override fun getInputErrors(copy: Boolean): InputErrors {

    val inputErrors: List<List<DenseNDArray>> = this.processor.getInputsErrors(copy = false)

    val contextErrors = List(size = inputErrors.size, init = {
      DenseNDArrayFactory.zeros(Shape(this.model.contextEncodingSize))
    })

    val rootErrors: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.model.contextEncodingSize))

    inputErrors.forEachIndexed { tokenIndex, (depErrors, govErrors) ->

      val tokenId: Int = this.dependencyTree.elements[tokenIndex]
      val depVector: DenseNDArray = contextErrors[tokenIndex]
      val govVector: DenseNDArray = this.dependencyTree.getHead(tokenId)?.let {
        contextErrors[this.dependencyTree.getPosition(it)]
      } ?: rootErrors

      depVector.assignSum(depErrors)
      govVector.assignSum(govErrors)
    }

    return InputErrors(rootErrors = rootErrors, contextErrors = contextErrors)
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the [Labeler] parameters
   */
  override fun getParamsErrors(copy: Boolean) = LabelerParams(this.processor.getParamsErrors(copy = copy))

  /**
   * Transform the array resulting from the prediction into a list of [ScoredGrammar].
   *
   * @return a list of [ScoredGrammar]
   */
  private fun DenseNDArray.toScoredGrammar(): List<ScoredGrammar> = (0 until this.length)
    .map { i -> ScoredGrammar(getGrammaticalConfiguration(i), score = this[i]) }
    .sortedWith(compareByDescending { it.score })

  /**
   * @param index a prediction index
   *
   * @return the grammatical configuration with the given [index]
   */
  private fun getGrammaticalConfiguration(index: Int): GrammaticalConfiguration =
    this.model.grammaticalConfigurations.getElement(index)!!

  /**
   * @param tokenId the id of a token of the input sentence
   * @param lss the latent syntactic structure of the input sentence
   *
   * @return the list of features that encode the given token
   */
  private fun extractFeatures(tokenId: Int, lss: LatentSyntacticStructure<*, *>): List<DenseNDArray> =
    listOf(
      lss.getContextVectorById(tokenId),
      this.dependencyTree.getHead(tokenId)?.let { lss.getContextVectorById(it) } ?: lss.virtualRoot
    )
}
