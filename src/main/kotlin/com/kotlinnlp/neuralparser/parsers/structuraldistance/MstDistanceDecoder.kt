package com.kotlinnlp.neuralparser.parsers.structuraldistance

import com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers.MSTHelper
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.utils.combine
import java.util.*
import kotlin.collections.HashSet

class MstDistanceDecoder (
    val distanceModel: StackedLayersParameters,
    val depthModel: StackedLayersParameters
) {

  /**
   * The structural distance predictor.
   */
  private val distancePredictor = StructuralDistancePredictor(
      model = this.distanceModel,
      useDropout = false)

  /**
   * The structural depth predictor.
   */
  private val depthPredictor = StructuralDepthPredictor(
      model = this.depthModel,
      useDropout = false)

  /**
   * Find the root node (the lowest depth node).
   *
   * @param depths the list of the depths of each node.
   *
   * @return the index of the root node.
   */
  private fun findRoot(depths: List<Double>): Int{
    var root = 0
    var min = Double.MAX_VALUE

    depths.forEachIndexed{ id, value -> if (value < min) {
      min = value
      root = id
    }
    }
    return root
  }

  /**
   * Create the list of the directed arcs, to build dependency tree.
   *
   * @param undirectedArcs the list of the scored undirected arcs.
   * @param depths the list of the depths of each node.
   *
   * @return a list of triples (governor, dependent, score) representing irected arcs
   */
  private fun findDirectedArcs(undirectedArcs: List<Triple<Int, Int, Double>>, depths: List<Double>): List<Triple<Int, Int, Double>>{
    val directedArcs = ArrayList<Triple<Int, Int, Double>>()
    val nodeSet: HashSet<Int> = HashSet<Int>()
    val visitedSet: HashSet<Int> = HashSet<Int>()
    val adjMatrix = Array(depths.size) { ArrayList<Pair<Int, Double>>() }
    val root = this.findRoot(depths)

    depths.forEachIndexed{i, depth -> nodeSet.add(i)}

    undirectedArcs.forEach{it ->
      adjMatrix[it.first].add(Pair(it.second, it.third))
      adjMatrix[it.second].add(Pair(it.first, it.third))
    }

    var currentNode = root
    visitedSet.add(root)

    while (nodeSet.size > 0) {

      adjMatrix[currentNode].forEach{it ->
        if (!visitedSet.contains(it.first)) {
          directedArcs.add(Triple(currentNode + 1, it.first + 1, it.second))
          visitedSet.add(it.first)
        }
      }

      nodeSet.remove(currentNode)
      visitedSet.forEach{it -> if (it in nodeSet) currentNode = it}
    }
    return directedArcs
  }

  /**
   * Decode the encoded representation into a list of arcs.
   *
   * @param ids the ids of the elements
   * @param contextVectors the encoded representation of the elements (aligned with the [ids])
   *
   * @return the list of arcs of the form <governor, dependent, score>
   */
  fun decode(ids: List<Int>, contextVectors: List<DenseNDArray>): List<Triple<Int, Int, Double>> {

    val pairs: List<Pair<Int, Int>> = contextVectors.indices.toList().combine()
    val distances: List<Double> = this.distancePredictor.forward(StructuralDistancePredictor.Input(hiddens = contextVectors, pairs = pairs))
    val depths: List<Double> = this.depthPredictor.forward(contextVectors)

    val mstHelper = MSTHelper(graphSize = ids.size, pairs = pairs, distances = distances)

    val mst = mstHelper.getMST()

    val arcs = this.findDirectedArcs(mst, depths)

    return arcs
  }

}