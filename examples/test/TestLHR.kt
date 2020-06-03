/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package test

import com.kotlinnlp.linguisticdescription.sentence.MorphoSynSentence
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyDictionary
import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.helpers.preprocessors.MorphoPreprocessor
import com.kotlinnlp.neuralparser.language.BaseSentence
import com.kotlinnlp.neuralparser.language.BaseToken
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRModel
import com.kotlinnlp.neuralparser.parsers.lhrparser.LHRParser
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.neuraltokenizer.Sentence
import com.kotlinnlp.neuraltokenizer.Token
import com.xenomachina.argparser.mainBody
import java.io.FileInputStream

/**
 * Test an [LHRModel] with texts read from the standard input.
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) = mainBody {

  val parsedArgs = CommandLineArguments(args)

  val parser: NeuralParser<*> = LHRParser(model = parsedArgs.parserModelPath.let {
    println("Loading model from '$it'.")
    NeuralParserModel.load(FileInputStream(it)) as LHRModel
  })
  val tokenizer = NeuralTokenizer(model = parsedArgs.tokenizerModelPath.let {
    println("Loading tokenizer model from '$it'...")
    NeuralTokenizerModel.load(FileInputStream(it))
  })
  val preprocessor = MorphoPreprocessor(dictionary = parsedArgs.morphoDictionaryPath.let {
    println("Loading serialized dictionary from '$it'...")
    MorphologyDictionary.load(FileInputStream(it))
  })

  while (true) {

    readInput()?.let { text ->

      val sentences: List<MorphoSynSentence> = tokenizer.tokenize(text).map {
        parser.parse(preprocessor.convert(it.toBaseSentence()))
      }

      sentences.forEachIndexed { i, s ->
        println("\nSentence #${i + 1}")
        println(s)
      }

    } ?: break
  }
}

/**
 * Read a text from the standard input.
 *
 * @return the string read or null if it was empty
 */
private fun readInput(): String? {

  print("\nInput text (empty to exit): ")

  return readLine()!!.trim().ifEmpty { null }
}

/**
 * @return a new base sentence built from this tokenizer sentence
 */
private fun Sentence.toBaseSentence() = BaseSentence(
  id = this.position.index,
  tokens = this.tokens.mapIndexed { i, it -> it.toBaseToken(id = i) },
  position = this.position)

/**
 * @param id the token ID
 *
 * @return a new base token built from this tokenizer token
 */
private fun Token.toBaseToken(id: Int) = BaseToken(id = id, position = this.position, form = this.form)
