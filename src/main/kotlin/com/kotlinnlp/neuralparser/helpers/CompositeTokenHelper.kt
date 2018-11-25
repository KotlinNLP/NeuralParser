package com.kotlinnlp.neuralparser.helpers

import com.kotlinnlp.dependencytree.DependencyTree
import com.kotlinnlp.linguisticdescription.GrammaticalConfiguration
import com.kotlinnlp.linguisticdescription.morphology.POS
import com.kotlinnlp.linguisticdescription.syntax.SyntacticType

/**
 * @param dependencyTree the dependency tree
 */
class CompositeTokenHelper(private val dependencyTree: DependencyTree) {

  /**
   * Get the ID of the governor of a component of a composite token.
   *
   * @param tokenId the ID of a parsing token
   * @param componentIndex the index of a component of the token
   * @param prevComponentId the ID assigned to the precedent component (null at the first component)
   *
   * @return the ID of the governor of the given component
   */
  fun getComponentGovernorId(tokenId: Int,
                             componentIndex: Int,
                             prevComponentId: Int?): Int? {

    val governorId: Int? = this.dependencyTree.getHead(tokenId)
    val config: GrammaticalConfiguration = this.dependencyTree.getConfiguration(tokenId)!!

    val isContin: Boolean = config.isContin()
    val isPrepArt: Boolean = config.isPrepArt()
    val isVerbEnclitic: Boolean = config.isVerbEnclitic()

    return when {
      componentIndex == 0 -> governorId
      isPrepArt && !isContin -> governorId
      isPrepArt && isContin -> this.dependencyTree.getMultiWordGovernorId(tokenId)
      isVerbEnclitic -> prevComponentId!!
      else -> null
    }
  }

  /**
   * Get the governor ID of a multi-word, given one of its tokens and going back through its ancestors in the dependency
   * tree.
   * Note: the governor of a multi-word is the governor of it first token.
   *
   * @param tokenId the id of a token that is part of a multi-word
   *
   * @return the governor id of the multi-word of which the given token is part of
   */
  private fun DependencyTree.getMultiWordGovernorId(tokenId: Int): Int? {

    var multiWordStartId: Int = this.getHead(tokenId)!!

    while (this.getConfiguration(multiWordStartId)!!.isContin())
      multiWordStartId = this.getHead(multiWordStartId)!!

    return this.getHead(multiWordStartId)
  }

  /**
   * @return true if this configuration defines the continuation of a multi-word, otherwise false
   */
  private fun GrammaticalConfiguration.isContin(): Boolean = this.components.any {
    it.syntacticDependency.isSubTypeOf(SyntacticType.Contin)
  }

  /**
   * @return true if this configuration defines a composite PREP + ART, otherwise false
   */
  private fun GrammaticalConfiguration.isPrepArt(): Boolean =
    this.components.size == 2 &&
      this.components[0].pos?.isSubTypeOf(POS.Prep) == true &&
      this.components[1].pos?.isSubTypeOf(POS.Art) == true

  /**
   * @return true if this configuration defines a composite VERB + PRON, otherwise false
   */
  private fun GrammaticalConfiguration.isVerbEnclitic(): Boolean =
    this.components.size >= 2 &&
      this.components[0].pos?.isSubTypeOf(POS.Verb) == true &&
      this.components.subList(1, this.components.size).all { it.pos?.isSubTypeOf(POS.Pron) == true }
}