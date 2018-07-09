/* Copyright 2018-present LHRParser Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package lhrparser.utils

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments for the transfer learning script.
 *
 * @param args the array of command line arguments
 */
class TransferLearningArgs(args: Array<String>)  {

  /**
   * The type of morphology encoding.
   */
  enum class MorphoEncodingType {
    POS_EMBEDDINGS, AMBIGUOUS_POS, MORPHO_EMBEDDINGS
  }

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The path of the file of the serialized model used as reference.
   */
  val referenceModelPath: String by parser.storing(
    "-r",
    "--reference-model-path",
    help="the path of the file of the serialized reference model"
  )

  /**
   * The number of training epochs (default = 30).
   */
  val epochs: Int by parser.storing(
    "-e",
    "--epochs",
    help="the number of training epochs (default = 30)"
  ) { toInt() }.default(30)

  /**
   * The maximum number of sentences to load for training (default unlimited)
   */
  val maxSentences: Int? by parser.storing(
    "-s",
    "--max-sentences",
    help="the maximum number of sentences to load for training (default unlimited)"
  ) { toInt() }.default { null }

  /**
   * The file path of the training set.
   */
  val trainingSetPath: String by parser.storing(
    "-t",
    "--training-set",
    help="the file path of the training set"
  )

  /**
   * The file path of the validation set.
   */
  val validationSetPath: String by parser.storing(
    "-v",
    "--validation-set",
    help="the file path of the validation set"
  )

  /**
   * The path of the file in which to save the serialized model.
   */
  val modelPath: String by parser.storing(
    "-m",
    "--model-path",
    help="the path of the file in which to save the serialized model"
  )

  /**
   * The file path of the pre-trained word embeddings.
   */
  val embeddingsPath: String? by parser.storing(
    "-w",
    "--trained-word-emb-path",
    help="the file path of the pre-trained word embeddings"
  ).default { null }

  /**
   * The size of the word embedding vectors.
   */
  val wordEmbeddingSize: Int by parser.storing(
    "--word-emb-size",
    help="the size of the word embedding vectors (default 150)"
  ){ toInt() }.default(150)

  /**
   * The word embeddings dropout coefficient.
   */
  val wordDropoutCoefficient: Double by parser.storing(
    "--word-dropout",
    help="the word embeddings dropout coefficient (default 0.25)"
  ){ toDouble() }.default(0.25)

  /**
   * The size of the part-of-speech embedding vectors.
   */
  val posEmbeddingSize: Int by parser.storing(
    "--pos-emb-size",
    help="the size of the part-of-speech embedding vectors (default 50)"
  ){ toInt() }.default(50)

  /**
   * The part-of-speech embeddings dropout coefficient.
   */
  val posDropoutCoefficient: Double by parser.storing(
    "--pos-dropout",
    help="the part-of-speech embeddings dropout coefficient (default 0.0)"
  ){ toDouble() }.default(0.0)

  /**
   * Whether to skip non-projective sentences.
   */
  val skipNonProjective: Boolean by parser.flagging(
    "--skip-non-projective",
    help="whether to skip non-projective sentences"
  )

  /**
   * The file path of the serialized morphology dictionary.
   */
  val morphoDictionaryPath: String? by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the serialized morphology dictionary"
  ).default { null }

  /**
   * The type of morphology encoding.
   */
  val morphoEncodingType: MorphoEncodingType by parser.mapping(
    "--morpho-pos-emb" to MorphoEncodingType.POS_EMBEDDINGS,
    "--morpho-ambiguous" to MorphoEncodingType.AMBIGUOUS_POS,
    "--morpho-emb" to MorphoEncodingType.MORPHO_EMBEDDINGS,
    help = "the type of morphology encoding (default --morpho-pos-emb)"
  ).default { MorphoEncodingType.POS_EMBEDDINGS }

  /**
   * Whether to do not show details about the training.
   */
  val quiet: Boolean by parser.flagging(
    "-q",
    "--quiet",
    help="whether to do not show details about the training "
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
