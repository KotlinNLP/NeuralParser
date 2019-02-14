package com.kotlinnlp.morphodisambiguator.language

import com.kotlinnlp.linguisticdescription.morphology.properties.MorphologyProperty
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.linguisticdescription.sentence.token.TokenIdentificable

data class MorphoToken (
    override val id: Int,
    override val form: String,
    val morphoProperties: List<List<MorphologyProperty?>>
) : FormToken, TokenIdentificable