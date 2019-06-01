package evaluation

import com.kotlinnlp.neuralparser.NeuralParser
import com.kotlinnlp.neuralparser.NeuralParserModel
import com.kotlinnlp.neuralparser.helpers.labelerselector.NoFilterSelector
import com.kotlinnlp.neuralparser.language.ParsingSentence
import com.kotlinnlp.neuralparser.language.ParsingToken
import com.kotlinnlp.neuralparser.parsers.structuraldistance.StructuralDistanceParser
import com.kotlinnlp.neuralparser.parsers.structuraldistance.StructuralDistanceParserModel
import com.kotlinnlp.neuraltokenizer.NeuralTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import java.io.File
import java.io.FileInputStream

/**
 * A simple example to parse the text entered from the standard input.
 *
 * Command line arguments:
 *   1. The file path of the tokenizer serialized model.
 *   2. The file path of the labeler serialized model.
 */
fun main(args: Array<String>) {

  require(args.size == 2) {
    "Required 2 arguments: <tokenizer_model_filename> <labeler_model_filename>."
  }

  val tokenizer: NeuralTokenizer = args[0].let {
    println("Loading tokenizer model from '$it'...")
    NeuralTokenizer(NeuralTokenizerModel.load(FileInputStream(File(it))))
  }

  println("Loading model \"${args[1]}\"...")

  val parser: NeuralParser<*> = StructuralDistanceParser(
    model = args[1].let {
      println("Loading model from '$it'.")
      NeuralParserModel.load(FileInputStream(File(it))) as StructuralDistanceParserModel
    })

  while (true) {

    val inputText = readValue()

    if (inputText.isEmpty()) break
    else {

      tokenizer.tokenize(inputText).forEach { sentence ->

        val parsingSentece = ParsingSentence(
          tokens = sentence.tokens.map { ParsingToken(id = it.position.index, form = it.form, position = it.position) },
          labelerSelector = NoFilterSelector,
          position = sentence.position
        )

        println(parser.parse(parsingSentece).tokens.joinToString(separator = "\n\n") { it.toString(prefix = "\t") })
      }
    }
  }

  println("\nExiting...")
}

/**
 * Read a value from the standard input.
 *
 * @return the string read
 */
private fun readValue(): String {

  print("\nLabel a text (empty to exit): ")

  return readLine()!!
}
