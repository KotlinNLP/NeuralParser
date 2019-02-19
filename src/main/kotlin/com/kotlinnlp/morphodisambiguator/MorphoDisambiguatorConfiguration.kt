package com.kotlinnlp.morphodisambiguator

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import java.io.Serializable

/**
 * The [MorphoDisambiguator] configuration.
 *
 * @property BiRNNHiddenSize the hidden size of the BIRNN encoder
 * @property BIRNNOutputSize the output size of the BIRNN encoder
 * @property parallelEncodersHiddenSize the hidden size of the sequence parallel encoder
 * @property parallelEncodersActivationFunction the activation function of the sequence parallel encoder
 */
data class MorphoDisambiguatorConfiguration(
    val BiRNNHiddenSize: Int,
    val BIRNNOutputSize: Int,
    val parallelEncodersHiddenSize: Int,
    val parallelEncodersActivationFunction: ActivationFunction
): Serializable {
  companion object {


    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

  }


}
