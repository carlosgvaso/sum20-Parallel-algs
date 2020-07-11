package hw2

import (
	"github.com/gonum/graph"
)

func BellmanFordImplChannel(u graph.Node, g graph.Graph) (path Shortest) {
	if !g.Has(u) {
		return Shortest{from: u}
	}
	var weight Weighting

	if wg, ok := g.(graph.Weighter); ok {
		weight = wg.Weight
	} else {
		weight = UniformCost(g)
	}

	nodes := g.Nodes()

	path = newShortestFrom(u, nodes)
	path.dist[path.indexOf[u.ID()]] = 0

	for i := 1; i < len(nodes); i++ {
		relax := false
		chanel := make(chan Shortest)

		for s, u := range nodes {
			for _, y := range g.From(u) {
				go func(u graph.Node, y graph.Node, chanel chan Shortest) {
					w, ok := weight(u, y)
					pan := false
					if !ok {
						pan = true
						//panic("belman-ford: negative edge weight")
					}
					if pan == false {
						d := path.indexOf[y.ID()]
						temp := path.dist[s] + w
						if temp < path.dist[d] {

							path.set(d, temp, s)
							relax = true
						}
						chanel <- path
					}

				}(u, y, chanel)

			}
			for _ = range g.From(u) {
				val, open := <-chanel
				if open == true {
					path = val
				} else {
					break
				}

			}
		}

		if relax == false {
			break
		}
	}
	for j, u := range nodes {
		for _, v := range g.From(u) {
			k := path.indexOf[v.ID()]
			w, ok := weight(u, v)
			if !ok {
				panic("bellman-ford: unexpected invalid weight")
			}
			if w < 0 {
				panic("belman-ford: negative edge weight")
			}
			if path.dist[j]+w < path.dist[k] {
				return path
			}
		}
	}

	return path
}
