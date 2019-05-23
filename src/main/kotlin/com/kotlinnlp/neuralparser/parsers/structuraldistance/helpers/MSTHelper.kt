package com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers

/**
 * The Minimum Spanning Tree helper. This class implements Prim algorithm.
 *
 * @param graphSize The number of nodes in the graph.
 * @param pairs The list of nodes pairs.
 * @param distances The list of distances between corresponding nodes pairs.
 */
class MSTHelper (
    private val graphSize: Int,
    private val pairs: List<Pair<Int, Int>>,
    private val distances: List<Double>
) {

  /**
   * The set of visited nodes.
   *
   */
  private val nodeSet: HashSet<Int> = HashSet()

  /**
   * The minimum cost of nodes not in the tree
   *
   */
  private val minCost = ArrayList<Double>(List(graphSize) {Double.MAX_VALUE})

  /**
   * The array that contains the minimum cost adjacent node.
   * The process starts from node 0, so node 0 is the virtual root of the tree.
   *
   */
  private val adjElements = ArrayList<Int>(List(graphSize) {-1})

  /**
   * The data structure representing the tree.
   *
   */
  private val adjMatrix = Array(this.graphSize) {Array(this.graphSize) {0.0} }

  /**
   * Set the adjacency matrix.
   */
  init{

    this.pairs.zip(distances) { pair, distance ->

      this.adjMatrix[pair.first][pair.second] = distance
      this.adjMatrix[pair.second][pair.first] = distance
    }
  }

  /**
   * Update the cost for the adjacent nodes of the [element]. I
   * If the cost is less than previous stored value, the array is updated.
   *
   * @param element a node of the graph
   */
  private fun updateCost(element: Int){

    val distances: Array<Double> = this.adjMatrix[element]

    for (i in distances.indices){
      if (!nodeSet.contains(i) && distances[i] < minCost[i]) {
        minCost[i] = distances[i]
        adjElements[i] = element
      }
    }
  }

  /**
   * Find the shortest path between visited nodes and other nodes.
   *
   */
  private fun findMin(): Pair<Int, Double>{

    var min: Double = Double.MAX_VALUE
    var argMin: Int = -1

    for (i in this.minCost.indices){
      if (!nodeSet.contains(i) && minCost[i] < min) {
        min = minCost[i]
        argMin = i
      }
    }

    return Pair(argMin, min)
  }

  /**
   * Return the Minimum Spanning Tree.
   *
   */
  fun getMST(): List<Triple<Int, Int, Double>>{

    val MST = ArrayList(emptyList<Triple<Int, Int, Double>>())
    var currentElement = 0

    nodeSet.add(currentElement)

    while (nodeSet.size != this.graphSize){

      this.updateCost(currentElement)
      val currentScore = this.findMin()
      currentElement = currentScore.first
      nodeSet.add(currentElement)
      MST.add(Triple(currentElement, adjElements[currentElement] ,currentScore.second))
    }

    return MST
  }
}