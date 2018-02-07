/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser

import com.kotlinnlp.simplednn.utils.Serializer
import java.io.InputStream
import java.io.OutputStream
import java.io.Serializable

/**
 * The serializable model of a [NeuralParser].
 */
abstract class NeuralParserModel : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Read a [NeuralParserModel] (serialized) from an input stream and decode it.
     *
     * @param inputStream the [InputStream] from which to read the serialized [NeuralParserModel]
     *
     * @return the [NeuralParserModel] read from [inputStream] and decoded
     */
    fun <ModelType: NeuralParserModel> load(inputStream: InputStream): ModelType = Serializer.deserialize(inputStream)
  }

  /**
   * Serialize this [NeuralParserModel] and write it to an output stream.
   *
   * @param outputStream the [OutputStream] in which to write this serialized [NeuralParserModel]
   */
  fun dump(outputStream: OutputStream) = Serializer.serialize(this, outputStream)
}
