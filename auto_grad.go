package gosor

import "fmt"

type GradientTracker struct {
	children []*GradientTracker
	gradient *Tensor
	gradFunc func() error
}

func (g *GradientTracker) Backward(l *Tensor) error {
	if g == nil {
		return fmt.Errorf("backward on tensor without gradient tracker")
	}

	// Topological sort of the graph.
	visited := make(map[*GradientTracker]bool)
	nodes := make([]*GradientTracker, 0)

	var buildTopo func(n *GradientTracker)
	buildTopo = func(n *GradientTracker) {
		if n == nil {
			return
		}
		if _, ok := visited[n]; ok {
			return
		}

		visited[n] = true
		for _, child := range n.children {
			if child != nil {
				buildTopo(child)
			}
		}
		nodes = append(nodes, n)
	}
	buildTopo(g)

	// Make sure gradient of the root node is set.
	g.gradient = l
	if g.gradient == nil {
		g.gradient, _ = New(WithValues(1))
	}

	// Go backwards and calculate the gradients.
	for i := len(nodes) - 1; i >= 0; i-- {
		if nodes[i].gradFunc != nil {
			err := nodes[i].gradFunc()
			if err != nil {
				return err
			}
		}
	}

	return nil
}
