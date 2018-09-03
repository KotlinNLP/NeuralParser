/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.neuralparser.parsers.lhrparser.helpers

import com.kotlinnlp.syntaxdecoder.utils.removeLast

/**
 * A helper that uses a beam of parallel states to finds the best configuration of a set of elements to which more
 * possible values can be assigned.
 *
 * @param valuesMap a map of element ids associated to their possible values, sorted by descending score
 * @param maxBeamSize the max number of parallel states that the beam supports
 * @param maxForkSize the max number of forks that can be generated from a state
 * @param maxIterations the max number of iterations of solving steps (it is the depth of beam recursion)
 */
internal abstract class BeamManager<ValueType: BeamManager.Value, StateType: BeamManager<ValueType, StateType>.State>(
  private val valuesMap: Map<Int, List<ValueType>>,
  private val maxBeamSize: Int = 5,
  private val maxForkSize: Int = 3,
  private val maxIterations: Int = 10
) {

  /**
   * An element of the state.
   *
   * @property id the id of this element
   * @property value the value of this element
   * @property index the index of the value within the possible values of this element
   */
  data class StateElement<T: BeamManager.Value>(val id: Int, val value: T, val index: Int)

  /**
   * The possible value of an element.
   */
  abstract class Value {

    /**
     * The score of this value.
     */
    abstract var score: Double
  }

  /**
   * The state that contains a configuration of values, one per element.
   *
   * @property elements the list of elements, each with a value assigned, sorted by diff score
   * @property forked whether this state has been already forked
   */
  abstract inner class State(val elements: List<StateElement<ValueType>>, var forked: Boolean = false) {

    /**
     * Whether this state is valid.
     */
    abstract var isValid: Boolean
      protected set

    /**
     * The score of the state.
     */
    abstract val score: Double

    /**
     * Fork this states into more, each built replacing the value of an element with the following in the
     * [scoresDiffMap].
     *
     * @return the list of new states forked from this
     */
    fun fork(): List<StateType> {

      val states: MutableList<StateType> = mutableListOf()

      run forkSteps@{
        this.elements.forEachIndexed { i, elm ->

          val possibleValues: List<ValueType> = valuesMap.getValue(elm.id)

          if (elm.index < possibleValues.lastIndex) {

            val newElements: List<StateElement<ValueType>> = this.elements
              .replace(i, elm.copy(index = elm.index + 1, value = possibleValues[elm.index + 1]))
              .sortedByAmbiguity()

            states.add(buildState(newElements))
          }

          if (states.size == maxForkSize) return@forkSteps
        }
      }

      this.forked = true

      return states
    }

    /**
     * @return the hash code of this state
     */
    override fun hashCode(): Int = this.elements.hashCode()

    /**
     * @param other any other object
     *
     * @return whether this state is equal to the given object
     */
    override fun equals(other: Any?): Boolean {

      if (this === other) return true
      if (javaClass != other?.javaClass) return false

      other as BeamManager<*, *>.State

      if (elements != other.elements) return false

      return true
    }

    /**
     * Replace an element of this list at a given index with another returning a new list.
     *
     * @param index the index of the replacement
     * @param elm the replacement
     *
     * @return a new list with the given element replaced
     */
    private fun <T> List<T>.replace(index: Int, elm: T): List<T> =
      this.mapIndexed { i, it ->  if (i == index) elm else it }

  }

  /**
   * The list of the best states found, sorted by descending score.
   */
  private val beam: MutableList<StateType> = mutableListOf()

  /**
   * A set of states already visited.
   */
  private val visitedStates: MutableSet<StateType> = mutableSetOf()

  /**
   * A remapping of the [valuesMap] to the differences between scores of consecutive values.
   */
  private val scoresDiffMap: Map<Int, List<Double>> = this.valuesMap.mapValues { (_, values) ->
    values.zipWithNext().map { it.first.score - it.second.score }
  }

  /**
   * Whether to prohibit invalid states into the beam.
   * It is false when the algorithm starts, in order to find a first valid state even if it is not present into the
   * first forks.
   * When a valid state is found, then it is set to true.
   */
  private var validStatesOnly = false

  /**
   * Find the best valid configuration of elements, with the highest global score.
   *
   * @return the state with the best configuration or null if no valid configuration has been found
   */
  fun findBestConfiguration(): StateType? {

    this.initBeam()

    run steps@{
      (0 until this.maxIterations).forEach { if (!this.solvingStep()) return@steps }
    }

    return this.beam.firstOrNull { it.isValid }
  }

  /**
   * Build a new state with the given elements.
   *
   * @param elements the elements that compose the building state
   *
   * @return a new state with the given elements
   */
  protected abstract fun buildState(elements: List<StateElement<ValueType>>): StateType

  /**
   * Initialize the beam with the first state (the one with highest score elements).
   */
  private fun initBeam() {

    val initState = this.buildState(
      this.valuesMap.map { (id, values) ->
        StateElement(id = id, index = 0, value = values.first()) // first = highest score value
      }.sortedByAmbiguity()
    )

    if (initState.isValid) this.validStatesOnly = true

    this.beam.add(initState)
  }

  /**
   * Let the beam go forward to the next step, forking each state into more.
   *
   * @return true if at least one new state has been added to the beam, otherwise false
   */
  private fun solvingStep(): Boolean {

    val forkedStates: List<StateType> = this.beam.filterNot { it.forked }.flatMap { it.fork() }

    if (!this.validStatesOnly && forkedStates.any { it.isValid }) {

      this.validStatesOnly = true

      this.beam.retainAll { it.isValid } // invalid states are no more allowed -> remove them from the beam
    }

    return this.getAllowedStates(forkedStates).map { this.addNewState(it) }.any { it }
  }

  /**
   * @param states a list of states
   *
   * @return the states that are allowed to enter the beam
   */
  private fun getAllowedStates(states: List<StateType>): List<StateType> =
    if (this.validStatesOnly) states.filter { it.isValid } else states

  /**
   * Add a new state (if possible) into the beam.
   *
   * @param state the new state to add
   *
   * @return true if a new state has been added, otherwise false
   */
  private fun addNewState(state: StateType): Boolean {

    if (state in this.visitedStates) return false

    val beamFull: Boolean = this.beam.size == this.maxBeamSize

    if (!beamFull || state.score > this.beam.last().score) {

      val index: Int = this.beam.indexOfFirst { it.score < state.score }.let { if (it >= 0) it else this.beam.size }

      if (beamFull) this.beam.removeLast()

      this.beam.add(index, state)
      this.visitedStates.add(state)

      return true
    }

    return false
  }

  /**
   * Note: a lower score difference means a greater ambiguity
   *
   * @return a new list with the values sorted by descending ambiguity.
   */
  private fun <T: Value> List<StateElement<T>>.sortedByAmbiguity(): List<StateElement<T>> =
    this.sortedBy { scoresDiffMap.getValue(it.id).getOrElse(it.index) { 1.0 } }
}
