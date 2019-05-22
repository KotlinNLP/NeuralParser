package com.kotlinnlp.neuralparser.parsers.structuraldistance

import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.neuralprocessor.feedforward.FeedforwardNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.removeAtIndexOfFirst
import com.kotlinnlp.utils.removeFrom

/**
 * Decode the dependencies using the Lower-Distance First (LDF) parsing strategy (Grella, 2019).
 */
class LowerDistanceFirstDecoder(
  val distanceModel: StackedLayersParameters,
  val depthModel: StackedLayersParameters
) {

  /**
   * The PendingElement represent a subtree with the left and the right spines.
   *
   * @param id the id
   * @param vector the vector representation
   * @param position the position in the list of elements
   */
  inner class PendingElement(val id: Int, val vector: DenseNDArray, val position: Int) {

    /**
     * The estimated depth of this element in the vector space of the sentence.
     */
    val depth: Double by lazy { this@LowerDistanceFirstDecoder.depthProcessor.forward(this.vector).expectScalar() }

    /**
     * Cache of the estimated distances of this element with another element in the sentence vector space.
     */
    private val distanceCache = mutableMapOf<PendingElement, Double>()

    /**
     * Predict the distance this element with the [other] in the sentence vector space.
     *
     * @param other an element
     *
     * @return the distance
     */
    fun distance(other: PendingElement): Double = this.distanceCache.getOrPut(other) {
      this@LowerDistanceFirstDecoder.distanceProcessor.forward(listOf(this.vector, other.vector)).expectScalar()
    }

    /**
     * The left spine (including itself for convenience)
     */
    val leftSpine: MutableList<PendingElement> = mutableListOf(this)

    /**
     * The right spine (including itself for convenience)
     */
    val rightSpine: MutableList<PendingElement> = mutableListOf(this)

    /**
     * @param k the index from which to add the [elements].
     * @param elements elements to add.
     */
    fun addToLeftSpine(k: Int, elements: List<PendingElement>) {
      this.leftSpine.removeFrom(k + 1).addAll(elements)
    }

    /**
     * @param k the index from which to add the [elements].
     * @param elements elements to add.
     */
    fun addToRightSpine(k: Int, elements: List<PendingElement>) {
      this.rightSpine.removeFrom(k + 1).addAll(elements)
    }
  }

  /**
   * The Action.
   *
   * @property pairIndex the index of the pair from which to extract the governor and the dependent
   * @property k the index of the left or right spine from which to add new elements
   * @property distance the predicted distance between the governor and the dependent
  */
  sealed class Action(val pairIndex: Int, val k: Int, val distance: Double) {

    /**
     * @param pairs the list of pairs from which to extract the governor and the dependent
     *
     * @return a triple of governor-id, dependent-id, distance
     */
    abstract fun perform(pairs: List<Pair<PendingElement, PendingElement>>): Triple<Int, Int, Double>

    /**
     * The ArcLeft action.
     *
     * @param pairIndex the index of the pair from which to extract the governor and the dependent
     * @param k the index of the left spine from which to add new elements
     * @param distance the predicted distance between the governor and the dependent
     */
    class ArcLeft(pairIndex: Int, k: Int, distance: Double) : Action(pairIndex, k, distance) {

      /**
       * @param pairs the list of pairs from which to extract the governor and the dependent
       *
       * @return a triple of governor-id, dependent-id, distance
       */
      override fun perform(pairs: List<Pair<PendingElement, PendingElement>>): Triple<Int, Int, Double> {

        val dep = pairs[this.pairIndex].first
        val gov = pairs[this.pairIndex].second

        gov.addToLeftSpine(this.k, dep.leftSpine)

        return Triple(gov.leftSpine[this.k].id, dep.id, this.distance)
      }
    }

    /**
     * The ArcRight action.
     *
     * @param pairIndex the index of the pair from which to extract the governor and the dependent
     * @param k the index of the right spine from which to add new elements
     * @param distance the predicted distance between the governor and the dependent
     */
    class ArcRight(pairIndex: Int, k: Int, distance: Double) : Action(pairIndex, k, distance) {

      /**
       * @param pairs the list of pairs from which to extract the governor and the dependent
       *
       * @return a triple of governor-id, dependent-id, distance
       */
      override fun perform(pairs: List<Pair<PendingElement, PendingElement>>): Triple<Int, Int, Double> {

        val dep = pairs[this.pairIndex].second
        val gov = pairs[this.pairIndex].first

        gov.addToRightSpine(this.k, dep.rightSpine)

        return Triple(gov.rightSpine[this.k].id, dep.id, this.distance)
      }
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

    val pendingList = ids.zip(vectors).mapIndexed { position, (id, v) ->
      PendingElement(id = id, vector = v, position = position)
    }.toMutableList()

    while (pendingList.size > 1) {

      val pairs = pendingList.zipWithNext()
      val possibleActions = this.generatePossibleActions(pairs)
      val bestAction = this.selectBestAction(possibleActions)
      val newArc = bestAction.perform(pairs)

      arcs.add(newArc)

      pendingList.removeAtIndexOfFirst { it.id == newArc.second } // remove the dependent
    }

    return arcs
  }

  /**
   * Generate the possible actions.
   *
   * @return the list of actions
   */
  private fun generatePossibleActions(pairs: List<Pair<PendingElement, PendingElement>>):
    List<LowerDistanceFirstDecoder.Action> {

    return pairs.foldIndexed(mutableListOf()) { i, acc, (first, second) ->

      if (first.depth < second.depth) { // first -> governor

        acc.add(LowerDistanceFirstDecoder.Action.ArcRight(
          pairIndex = i,
          k = 0,
          distance = first.distance(second)))

        first.rightSpine.forEachIndexed { k, element ->

          if (k > 0 && second.depth > element.depth) { // filter impossible attachment

            acc.add(LowerDistanceFirstDecoder.Action.ArcRight(
              pairIndex = i,
              k = k,
              distance = element.distance(second)))
          }
        }

      } else { // second -> governor

        acc.add(LowerDistanceFirstDecoder.Action.ArcLeft(
          pairIndex = i,
          k = 0,
          distance = first.distance(second)))

        second.leftSpine.forEachIndexed { k, element ->

          if (k > 0 && first.depth > element.depth) { // filter impossible attachment

            acc.add(LowerDistanceFirstDecoder.Action.ArcLeft(
              pairIndex = i,
              k = k,
              distance = element.distance(first)))
          }
        }
      }

      acc
    }
  }

  /**
   * Select the action with the lowest distance.
   *
   * @return the best action
   */
  private fun selectBestAction(actions: List<LowerDistanceFirstDecoder.Action>) =
    actions.sortedBy { it.distance }.first()
}