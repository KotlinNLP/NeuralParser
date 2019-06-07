package parsers.structuraldistance

import com.kotlinnlp.neuralparser.parsers.structuraldistance.helpers.DirectedGraphHelper
import org.jetbrains.spek.api.Spek
import org.jetbrains.spek.api.dsl.describe
import org.jetbrains.spek.api.dsl.it
import org.jetbrains.spek.api.dsl.on

/**
 *
 */
class DirectedGraphHelperSpec : Spek({

  describe("The Directed graph Helper") {

    on("find cycles") {
      val graph = DirectedGraphUtils.getGraph1()


      it("should match the expected output") {
        val cycles =DirectedGraphHelper.findCycles(graph)

        print(cycles)
      }
    }

  }

})
