/* Copyright 2018-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package constraints

import com.kotlinnlp.neuralparser.constraints.Constraint
import constraints.utils.loadConstraints

/**
 * Read constraints from a given JSON file and print them to the standard output.
 * The JSON file must contain an array of objects, each defining a constraint.
 *
 * Command line arguments:
 *  1. The path of the input JSON file.
 */
fun main(args: Array<String>) {

  val constraints: List<Constraint> = loadConstraints(args[0])

  println("${constraints.size} constraints loaded:")

  constraints.forEach {
    println("- ${it::class.simpleName}: $it")
  }
}
