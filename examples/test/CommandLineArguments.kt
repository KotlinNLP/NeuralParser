/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package test

import com.xenomachina.argparser.ArgParser

/**
 * The interpreter of command line arguments for the evaluation script.
 *
 * @param args the array of command line arguments
 */
class CommandLineArguments(args: Array<String>) {

  /**
   * The parser of the string arguments.
   */
  private val parser = ArgParser(args)

  /**
   * The file path of the serialized parser model.
   */
  val parserModelPath: String by parser.storing(
    "-m",
    "--model",
    help="the file path of the serialized parser model"
  )

  /**
   * The file path of the serialized tokenizer model.
   */
  val tokenizerModelPath: String by parser.storing(
    "-t",
    "--tokenizer",
    help="the file path of the serialized tokenizer model"
  )

  /**
   * The file path of the serialized morphology dictionary.
   */
  val morphoDictionaryPath: String by parser.storing(
    "-d",
    "--dictionary",
    help="the file path of the serialized morphology dictionary"
  )

  /**
   * Force parsing all arguments (only read ones are parsed by default).
   */
  init {
    parser.force()
  }
}
