package hw2

import (
	"github.com/gonum/graph"
	"github.com/gonum/graph/path"
)

func BellmanFord(s graph.Node, g graph.Graph) path.Shortest {
	// Your code goes here.
	return path.Shortest{}
}

func DeltaStep(s graph.Node, g graph.Graph) path.Shortest {
	// Your code goes here.
	return path.Shortest{}
}

// Runs dijkstra from gonum to make sure that the tests are correct.
func Dijkstra(s graph.Node, g graph.Graph) path.Shortest {
	return path.DijkstraFrom(s, g)
}
