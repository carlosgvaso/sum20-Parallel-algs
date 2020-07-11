package hw2

import (
	"fmt"
	"math"
	"sync"

	"github.com/gonum/graph"
)

// Global variables
// ----------------

// delta is the delta parameter of Delta-Stepping algorithm
var delta int = 10

// DEBUG prints debugging output, if set to true
var DEBUG bool = false

// BLock is the B mutex
var BLock sync.RWMutex

// dLock is the d mutex
var dLock sync.RWMutex

// Funtions
// --------

/* DeltaStepFrom returns a shortest-path tree for a shortest path from u to all
 * nodes in the graph g. If the graph does not implement graph.Weighter,
 * UniformCost is used. DeltaStepFrom will panic if g has a u-reachable negative
 * edge weight.
 *
 * Delta-Stepping algorithm pseudocode:
 *
 * var:
 * 	   d: array[0..n-1] of int initially for-all i : d[i] = Inf
 *     req, S: set of (int, int) (implement as a set of Edge types)
 *     B: sequence of buckets (sets) (implement as a dynamic array of sets)
 * relax(0, 0);
 * int i := 0;
 * while B != EmptySequence
 *     S := {}
 *     while B[i] != {}
 *         req := {(v, d[u] + w[u, v] | u ELEMENT_OF B[i] AND (u, v) ELEMENT_OF E_light}
 *         S := S UNION B[i]
 *         B[i] := {}
 *         forall (v, d[u] + w[u, v]) ELEMENT_OF req in par do: relax(v, d[u] + w[u, v])
 *     endwhile
 *     req := {(v, d[u] + w[u, v] | u ELEMENT_OF S AND (u, v) ELEMENT_OF E_heavy}
 *     forall (v, d[u] + w[u, v]) ELEMENT_OF req in par do: relax(v, d[u] + w[u, v])
 *     i := i + 1
 * endwhile
 *
 * function relax(int v, int c)
 *     if c < d[v]
 *         d[v] := c
 *         move node v to the bucket B[c/delta]
 * endfunction
 */
func DeltaStepFrom(s graph.Node, g graph.Graph) Shortest {
	if !g.Has(s) {
		if DEBUG {
			fmt.Printf("return: s is not in graph\n")
		}
		return Shortest{from: s}
	}

	var weight Weighting
	if wg, ok := g.(graph.Weighter); ok {
		weight = wg.Weight
	} else {
		weight = UniformCost(g)
	}

	var waitGroup sync.WaitGroup // Wait group to synchronize parallel goroutines
	nodes := g.Nodes()
	d := newShortestFrom(s, nodes) // This will be used instead of an array
	var n int = len(nodes)         // Total number of nodes

	// Initialize req and S sets, and bucket sequence
	req := makeSetTuple()
	S := makeSetTuple()
	B := make(map[int]SetTuple)
	var idx int = 0

	if DEBUG {
		fmt.Printf("DeltaStep: delta = %d\n", delta)
		fmt.Printf("DeltaStep: s.ID(): %d\n", s.ID())
		fmt.Printf("DeltaStep: n = %d\n", n)
		fmt.Printf("DeltaStep: d[%d] = [", len(d.dist))
		for i, v := range nodes {
			if i == 0 {
				fmt.Printf("(%d)%f", v.ID(), d.dist[d.indexOf[v.ID()]])
			} else {
				fmt.Printf(", (%d)%f", v.ID(), d.dist[d.indexOf[v.ID()]])
			}
		}
		fmt.Printf("]\n")
		fmt.Printf("DeltaStep: req = ")
		req.print()
		fmt.Printf("\n")
		fmt.Printf("DeltaStep: S = ")
		S.print()
		fmt.Printf("\n")
		fmt.Printf("DeltaStep: B = [ ")
		for i, set := range B {
			if i == 0 {
				set.print()
			} else {
				fmt.Printf(", ")
				set.print()
			}
		}
		fmt.Printf(" ]\n")
		fmt.Printf("DeltaStep: idx = %d\n", idx)
	}

	/* relax(0, 0);
	 *
	 * Since newShortestFrom() already initializes the source node to 0 and the
	 * other nodes to +Inf, we don't need to relax the source node. We just need
	 * to add it to the first bucket, B[0].
	 */
	B[0] = makeSetTuple()
	B[0].add(s.ID(), s.ID(), float64(0.0))
	if DEBUG {
		fmt.Printf("DeltaStep: B = [ ")
		for i, set := range B {
			if i == 0 {
				set.print()
			} else {
				fmt.Printf(", ")
				set.print()
			}
		}
		fmt.Printf(" ]\n")
		fmt.Printf("DeltaStep: idx = %d\n", idx)
	}

	// while B != EmptySequence
	for len(B) > 0 {
		// S := {}
		S = makeSetTuple()
		if DEBUG {
			fmt.Printf("DeltaStep: S = ")
			S.print()
			fmt.Printf("\n")
		}

		// while B[i] != {}
		if DEBUG {
			fmt.Printf("DeltaStep: B = [ ")
			for i, set := range B {
				if i == 0 {
					set.print()
				} else {
					fmt.Printf(", ")
					set.print()
				}
			}
			fmt.Printf(" ]\n")
			fmt.Printf("DeltaStep: idx = %d\n", idx)
		}
		for B[idx].size() > 0 {
			// req := {(v, d[u] + w[u, v] | u ELEMENT_OF B[i] AND (u, v) ELEMENT_OF E_light}
			req = findLightEdges(B[idx], &d, &g, &weight)
			if DEBUG {
				fmt.Printf("DeltaStep: req = ")
				req.print()
				fmt.Printf("\n")
			}

			// S := S UNION B[i]
			S.union(B[idx])
			if DEBUG {
				fmt.Printf("DeltaStep: S = ")
				S.print()
				fmt.Printf("\n")
			}

			// B[i] := {}
			B[idx] = makeSetTuple()

			// forall (v, d[u] + w[u, v]) ELEMENT_OF req in par do: relax(v, d[u] + w[u, v])
			for tuple := range req {
				waitGroup.Add(1)
				go relax(tuple, B, idx, &d, &waitGroup)
			}
			waitGroup.Wait()

		}

		// req := {(v, d[u] + w[u, v] | u ELEMENT_OF S AND (u, v) ELEMENT_OF E_heavy}
		if DEBUG {
			fmt.Printf("DeltaStep: S = ")
			S.print()
			fmt.Printf("\n")
		}
		req = findHeavyEdges(S, &d, &g, &weight)
		if DEBUG {
			fmt.Printf("DeltaStep: req = ")
			req.print()
			fmt.Printf("\n")
		}

		// forall (v, d[u] + w[u, v]) ELEMENT_OF req in par do: relax(v, d[u] + w[u, v])
		for tuple := range req {
			waitGroup.Add(1)
			go relax(tuple, B, idx, &d, &waitGroup)
		}
		waitGroup.Wait()

		// Delete bucket now that we processed all its nodes
		delete(B, idx)

		// i := i + 1
		idx++

		if DEBUG {
			fmt.Printf("DeltaStep: len(B) = %d\n", len(B))
		}
	}

	if DEBUG {
		PrintResult(d, g)
	}
	return d
}

/* relax is the edge relaxation function.
 *
 * param	tuple	SetTuple object containing the edge to relax
 * param	B		Bucket sequence object
 * param	idx		Key of the current bucket being explored
 * param	d		Pointer to current Shortest struct containing the distances
 * param	waitGroup	Pointer to sync.WaitGroup object for parallelization
 */
func relax(tuple Tuple, B map[int]SetTuple, idx int, d *Shortest, waitGroup *sync.WaitGroup) {
	defer waitGroup.Done()

	dLock.RLock()
	u := d.indexOf[tuple.u] // Used to access d.dist[]
	v := d.indexOf[tuple.v] // Used to access d.dist[]
	dLock.RUnlock()
	c := tuple.dist

	// if c < d[v]
	if c < d.dist[v] {
		// d[v] := c
		dLock.Lock()
		d.set(v, c, u)
		dLock.Unlock()

		// move v to the bucket B[c/delta]
		BLock.Lock()
		dLock.RLock()
		B[idx].delete(tuple.u, tuple.v, (*d).dist[v]) // Remove from initial bucket
		dLock.RUnlock()

		// Check new if bucket exists, or create it
		newBucketIdx := int(math.Floor(c / float64(delta)))
		_, ok := B[newBucketIdx]
		if !ok {
			B[newBucketIdx] = makeSetTuple()
		}

		B[newBucketIdx].add(tuple.u, tuple.v, c) // Add to new bucket
		BLock.Unlock()
	}
}

/* findLightEdges finds all light edges in the provided bucket in parallel.
 *
 * The input set is of the format: { (u1, v1, d1), (u2, v2, d2) ... }. The u1,
 * u2, ... represent the index of the from nodes in the set, the v1, v2, ...
 * represent the index of the to nodes in the set, and the d1, d2, ... represent
 * the current distance of the to nodes from the source node passing through the
 * (u1,v1), (u2,v2), ... edges.
 *
 * param	set	Set of type SetTuple to search
 * param	d	Pointer to current Shortest struct containing the distances
 * param	g	Pointer to curent graph
 * return	Set of type SetTuple containing all light edges found in input set
 *
 * TODO: Parallelize
 */
func findLightEdges(set SetTuple, d *Shortest, g *graph.Graph, weight *Weighting) SetTuple {
	edgesLight := makeSetTuple()

	// Iterate over all nodes in the set
	for tuple := range set {
		// Get all the nodes that can be reached from the tuple from node (u)
		u := (*g).Nodes()[tuple.u]
		vs := (*g).From(u)

		// Iterate over all reachable nodes
		for _, v := range vs {
			// Get edge weight
			//w := (*g).Edge(u, v).Weight()
			w, ok := (*weight)(u, v)

			if !ok {
				w = float64(1)
			}

			// Save the light edges
			if w <= float64(delta) {
				edgesLight.add(u.ID(), v.ID(), (*d).dist[(*d).indexOf[u.ID()]]+w)
			}
		}

	}
	return edgesLight
}

/* findHeavyEdges finds all heavy edges in the provided bucket in parallel.
 *
 * The input set is of the format: { (u1, v1, d1), (u2, v2, d2) ... }. The u1,
 * u2, ... represent the index of the from nodes in the set, the v1, v2, ...
 * represent the index of the to nodes in the set, and the d1, d2, ... represent
 * the current distance of the to nodes from the source node passing through the
 * (u1,v1), (u2,v2), ... edges.
 *
 * param	set	Set of type SetTuple to search
 * param	d	Pointer to current Shortest struct containing the distances
 * param	g	Pointer to curent graph
 * return	Set of type SetTuple containing all heavy edges found in input set
 *
 * TODO: Parallelize
 */
func findHeavyEdges(set SetTuple, d *Shortest, g *graph.Graph, weight *Weighting) SetTuple {
	edgesHeavy := makeSetTuple()

	// Iterate over all nodes in the set
	// TODO: Do in parallel
	for tuple := range set {
		// Get all the nodes that can be reached from the tuple node
		u := (*g).Nodes()[tuple.v]
		vs := (*g).From(u)

		// Iterate over all reachable nodes
		// TODO: Do in parallel (?)
		for _, v := range vs {
			// Get edge weight
			//w := (*g).Edge(u, v).Weight()
			w, ok := (*weight)(u, v)

			if !ok {
				w = float64(1)
			}

			// Save the heavy edges
			if w > float64(delta) {
				edgesHeavy.add(u.ID(), v.ID(), (*d).dist[(*d).indexOf[u.ID()]]+w)
			}
		}

	}
	return edgesHeavy
}

// Set implementation of Tuple type elements.

// Tuple Struct for the tuple type of format: (u, v, dist)
type Tuple struct {
	u    int     // From node ID in graph
	v    int     // To node ID in graph
	dist float64 // Distance from source node to To node passing through the (u, v) edge
}

// SetTuple Type for the Tuple type set of format: { (u1, v1, dist1), ...}
type SetTuple map[Tuple]bool

func makeSetTuple() SetTuple {
	return make(SetTuple)
}

func (set SetTuple) add(u int, v int, dist float64) bool {
	tuple := Tuple{u, v, dist}
	_, found := set[tuple]
	set[tuple] = true
	return !found //False if it existed already
}

func (set SetTuple) delete(u int, v int, dist float64) {
	tuple := Tuple{u, v, dist}
	delete(set, tuple)
}

func (set SetTuple) union(set2 SetTuple) {
	for k := range set2 {
		set.add(k.u, k.v, k.dist)
	}
}

func (set SetTuple) size() int {
	return len(set)
}

func (set SetTuple) contains(u int, v int, dist float64) bool {
	tuple := Tuple{u, v, dist}
	_, found := set[tuple]
	return found
}

func (set SetTuple) print() {
	var i int = 0
	fmt.Printf("{ ")
	for tuple := range set {
		if i == 0 {
			fmt.Printf("( u:%d, v:%d, dist:%f )", tuple.u, tuple.v, tuple.dist)
		} else {
			fmt.Printf(", ( u:%d, v:%d, dist:%f )", tuple.u, tuple.v, tuple.dist)
		}
		i++
	}
	fmt.Printf(" }")
}

// PrintResult prints the resulting Shortest struct.
func PrintResult(d Shortest, g graph.Graph) {
	fmt.Printf("\nSource: %d\n", d.From().ID())
	nodes := g.Nodes()
	for _, node := range nodes {
		path, dist := d.To(node)
		fmt.Printf("Node: %d, Dist: %f", node.ID(), dist)
		for i, n := range path {
			if i == 0 {
				fmt.Printf(", Path: [ %d", n.ID())
			} else {
				fmt.Printf(", %d", n.ID())
			}
		}
		fmt.Printf(" ]\n")
	}
	fmt.Printf("\n")
}

/*
type distanceNode struct {
	node graph.Node
	dist float64
}

// priorityQueue implements a no-dec priority queue.
type priorityQueue []distanceNode

func (q priorityQueue) Len() int            { return len(q) }
func (q priorityQueue) Less(i, j int) bool  { return q[i].dist < q[j].dist }
func (q priorityQueue) Swap(i, j int)       { q[i], q[j] = q[j], q[i] }
func (q *priorityQueue) Push(n interface{}) { *q = append(*q, n.(distanceNode)) }
func (q *priorityQueue) Pop() interface{} {
	t := *q
	var n interface{}
	n, *q = t[len(t)-1], t[:len(t)-1]
	return n
}
*/
