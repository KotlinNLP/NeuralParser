/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package morphologicaldisambiguator.evaluation

import com.xenomachina.argparser.ArgParser
import com.xenomachina.argparser.default

/**
 * The interpreter of command line arguments for the evaluation script.
 *
 * @param args the array of command line arguments
 */
@Suppress("unused")
class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The file path of the serialized model.
   */
  val modelPath: String by parser.storing(
      "-m",
      "--model-path",
      help="the file path of the serialized model"
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
   * The file path of LSS encoder model
   */
  val lssModelPath: String? by parser.storing(
      "--lss-model-path",
      help="the model path of the LSS encoder model"
  ).default { null }

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
