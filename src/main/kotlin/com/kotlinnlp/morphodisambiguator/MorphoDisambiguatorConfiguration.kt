package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction

/**
 * The [MorphoDisambiguator] configuration.
 *
 * @property tokensEncoderParams the parameters of the tokens encoder
 * @property contextEncoderParams the parameters of the context encoder
 * @property headsEncoderParams the parameters of the heads encoder
 */
data class MorphoDisambiguatorConfiguration(
    val BiRNNHiddenSize: Int,
    val BIRNNOutputSize: Int,
    val parallelEncodersHiddenSize: Int,
    val parallelEncodersActivationFunction: ActivationFunction
)
