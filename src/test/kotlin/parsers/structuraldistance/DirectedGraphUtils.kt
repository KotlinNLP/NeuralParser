package parsers.structuraldistance

import com.kotlinnlp.neuralparser.parsers.structuraldistance.StructuralDistancePredictor
import com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers.DirectedGraph
import com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers.DirectedGraphHelper

object DirectedGraphUtils {

  /**
   *
   */
  fun getGraph1(): DirectedGraph{

    val pairs: ArrayList<Pair<Int, Int>> = ArrayList()
    val distances: ArrayList<Double> = ArrayList()
    val depths: ArrayList<Double> = ArrayList()

    pairs.add(Pair(0, 1))
    pairs.add(Pair(0, 2))
    pairs.add(Pair(1, 2))
    pairs.add(Pair(5, 1))
    pairs.add(Pair(5, 4))
    pairs.add(Pair(3, 4))
    pairs.add(Pair(5, 6))
    pairs.add(Pair(6, 7))

    distances.add(0.4)
    distances.add(0.3)
    distances.add(0.5)
    distances.add(2.5)
    distances.add(1.2)
    distances.add(1.3)
    distances.add(0.3)
    distances.add(0.4)

    depths.add(0.5)
    depths.add(0.6)
    depths.add(0.7)
    depths.add(10.5)
    depths.add(8.5)
    depths.add(7.4)
    depths.add(7.3)
    depths.add(7.2)

    return DirectedGraph(graphSize = 8, pairs = pairs, distances = distances, depths = depths, penalty = true)
  }

}