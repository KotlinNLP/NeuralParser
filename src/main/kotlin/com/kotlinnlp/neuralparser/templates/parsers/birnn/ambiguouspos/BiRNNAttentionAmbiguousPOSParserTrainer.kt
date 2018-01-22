package com.kotlinnlp.neuralparser.templates.parsers.birnn.ambiguouspos

import com.kotlinnlp.neuralparser.helpers.Validator
import com.kotlinnlp.neuralparser.templates.featuresextractor.AttentionFeaturesExtractor
import com.kotlinnlp.neuralparser.templates.inputcontexts.TokensAmbiguousPOSContext
import com.kotlinnlp.neuralparser.utils.items.DenseItem
import com.kotlinnlp.syntaxdecoder.modules.actionserrorssetter.ActionsErrorsSetter
import com.kotlinnlp.syntaxdecoder.modules.supportstructure.DecodingSupportStructure
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.Features
import com.kotlinnlp.syntaxdecoder.modules.featuresextractor.features.FeaturesErrors
import com.kotlinnlp.syntaxdecoder.transitionsystem.Transition
import com.kotlinnlp.syntaxdecoder.transitionsystem.oracle.OracleFactory
import com.kotlinnlp.syntaxdecoder.transitionsystem.state.State

/**
 * The training helper of the BiRNN Attention Parser.
 *
 * @param neuralParser a neural parser
 * @param epochs the number of training epochs
 * @param batchSize the size of the batches of sentences
 * @param minRelevantErrorsCountToUpdate the min number of relevant errors needed to update the neural parser
 *                                       (default = 1)
 * @param validator the validation helper (if it is null no validation is done after each epoch)
 * @param modelFilename the name of the file in which to save the best trained model
 * @param verbose a Boolean indicating if the verbose mode is enabled (default = true)
 */
class BiRNNAttentionAmbiguousPOSParserTrainer<
  StateType : State<StateType>,
  TransitionType : Transition<TransitionType, StateType>,
  FeaturesErrorsType: FeaturesErrors,
  FeaturesType : Features<FeaturesErrorsType, *>,
  SupportStructureType : DecodingSupportStructure,
  ModelType: BiRNNAmbiguousPOSParserModel
  >(
  neuralParser: BiRNNAmbiguousPOSParser<StateType, TransitionType, FeaturesErrorsType, FeaturesType,
    SupportStructureType, ModelType>,
  actionsErrorsSetter: ActionsErrorsSetter<StateType, TransitionType, DenseItem, TokensAmbiguousPOSContext>,
  oracleFactory: OracleFactory<StateType, TransitionType>,
  epochs: Int,
  batchSize: Int,
  minRelevantErrorsCountToUpdate: Int = 1,
  validator: Validator?,
  modelFilename: String,
  verbose: Boolean = true
) :
  BiRNNAmbiguousPOSParserTrainer<
    StateType,
    TransitionType,
    FeaturesErrorsType,
    FeaturesType,
    SupportStructureType,
    ModelType>
  (
    neuralParser = neuralParser,
    actionsErrorsSetter = actionsErrorsSetter,
    oracleFactory = oracleFactory,
    epochs = epochs,
    batchSize = batchSize,
    minRelevantErrorsCountToUpdate = minRelevantErrorsCountToUpdate,
    validator = validator,
    modelFilename = modelFilename,
    verbose = verbose
  ) {

  /**
   * Callback called after the learning of a sentence.
   */
  override fun afterSentenceLearning(context: TokensAmbiguousPOSContext) {

    (this.neuralParser.syntaxDecoder.featuresExtractor as AttentionFeaturesExtractor).afterSentenceLearning()

    this.propagateErrors(context = context)
  }
}