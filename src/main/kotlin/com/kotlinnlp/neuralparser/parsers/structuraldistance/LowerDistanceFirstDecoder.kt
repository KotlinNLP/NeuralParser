package com.kotlinnlp.neuralparser.parsers.structuraldistance

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Decode the dependencies using the Lower-Distance First (LDF) parsing strategy (Grella, 2019).
 */
class LowerDistanceFirstDecoder(
  val distanceModel: StackedLayersParameters,
  val depthModel: StackedLayersParameters
) {

  /**
   * The Element.
   *
   * @param id the id
   * @param vector the vector representation
   */
  inner class Element(val id: Int, val vector: DenseNDArray) {

    /**
     * The estimated depth of this element in the sentence vector space.
     */
    val depth: Double by lazy { this@LowerDistanceFirstDecoder.depthProcessor.forward(this.vector).expectScalar() }

    /**
     * Cache of the estimated distances of this element with another element in the sentence vector space.
     */
    private val distanceCache = mutableMapOf<Element, Double>()

    /**
     * Predict the distance this element with the [other] in the sentence vector space.
     *
     * @param other an element
     *
     * @return the distance
     */
    fun distance(other: Element): Double = this.distanceCache[other]
      ?: this@LowerDistanceFirstDecoder.distanceProcessor.forward(listOf(this.vector, other.vector)).expectScalar().also {
        this.distanceCache[other] = it
      }
  }

  /**
   * The processor to predict the 'tree-depth'.
   */
  private val depthProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = this.depthModel,
    useDropout = false,
    propagateToInput = false)

  /**
   * The processor to predict the 'tree-distance'.
   */
  private val distanceProcessor = FeedforwardNeuralProcessor<DenseNDArray>(
    model = this.distanceModel,
    useDropout = false,
    propagateToInput = false)

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param ids the ids of the elements
   * @param vectors the encoded representation of the elements (aligned with the [ids])
   *
   * @return the list of arcs of the form <governor, dependent, score>
   */
  fun decode(ids: List<Int>, vectors: List<DenseNDArray>): List<Triple<Int, Int, Double>> {

    val arcs = mutableListOf<Triple<Int, Int, Double>>()
    val elements = ids.zip(vectors).map { (id, v) -> Element(id = id, vector = v) }
    val pendingList = elements.toMutableList()

    while (pendingList.size > 1) {

      val pairs = pendingList.zipWithNext()

      val (first, second, distance) = this.selectBest(pairs)

      val dep: Element
      val gov: Element

      if (first.depth > second.depth) {
        dep = first
        gov = second
      } else {
        dep = second
        gov = first
      }

      arcs.add(Triple(gov.id, dep.id, distance))

      pendingList.removeAtIndexOfFirst { it.id == dep.id }
    }

    return arcs
  }

  /**
   * Select the pair with the lowest distance.
   *
   * @return a triple <first, second, distance>
   */
  private fun selectBest(pairs: List<Pair<Element, Element>>): Triple<Element, Element, Double> = pairs
    .sortedBy { (a, b) -> a.distance(b) }
    .first()
    .let { (a, b) -> Triple(a, b, a.distance(b)) }

  /**
   * Removes an element at the position of the first element that matches the given [predicate].
   *
   * @return the element that has been removed.
   */
  private fun <T> MutableList<T>.removeAtIndexOfFirst(predicate: (T) -> Boolean)
    = this.removeAt(this.indexOfFirst(predicate))
}