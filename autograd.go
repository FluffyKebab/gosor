package gosor

import "fmt"

type GradFunc func() ([]*Tensor, error)

type GradientTracker struct {
	children  []*GradientTracker
	gradient  *Tensor
	gradFunc  GradFunc
	isNotLeaf bool
}

func (g *GradientTracker) Gradient() (*Tensor, error) {
	if g.gradient == nil {
		return nil, fmt.Errorf("gradient not calculated")
	}
	return g.gradient, nil
}

func (g *GradientTracker) ResetGradient() {
	g.gradient = nil
}

func (g *GradientTracker) Backward(previousGradient *Tensor) (err error) {
	if g == nil {
		return fmt.Errorf("backwards on tensor without gradient tracker")
	}
	if previousGradient == nil {
		previousGradient = Wrap(New(WithValues(1))).MustValue()
	}

	if g.gradient == nil {
		g.gradient, err = New(WithSize(previousGradient.sizes...))
		if err != nil {
			return err
		}
	}

	_, err = AddInto(g.gradient, g.gradient, previousGradient)
	if err != nil {
		return err
	}

	if g.gradFunc == nil {
		return
	}
	localGradients, err := g.gradFunc()
	if err != nil {
		return fmt.Errorf("calculating local gradient: %w", err)
	}
	if len(localGradients) != len(g.children) {
		return fmt.Errorf("wrong amount tensors returned from grad func, should be one for each child.")
	}

	globalGradient := make([]*Tensor, 0, len(localGradients))
	for i := 0; i < len(localGradients); i++ {
		if localGradients[i] == nil {
			continue
		}
		curGlobalGrad, err := Mul(localGradients[i], previousGradient)
		if err != nil {
			return fmt.Errorf("multiplication of local and global gradient: %w", err)
		}
		globalGradient = append(globalGradient, curGlobalGrad)
	}

	for i := 0; i < len(globalGradient); i++ {
		if globalGradient[i] != nil && g.children[i] != nil {
			err := g.children[i].Backward(globalGradient[i])
			if err != nil {
				return err
			}
		}
	}

	return
}

func addGradientTracker(res *Tensor, children []*Tensor, gradFunc GradFunc) {
	if res == nil || res.GradientTracker == nil {
		return
	}
	if res.isNotLeaf {
		childrenTrackers := make([]*GradientTracker, len(children))
		shouldHaveTracker := false
		for i := 0; i < len(children); i++ {
			if children[i].GradientTracker != nil {
				childrenTrackers[i] = children[i].GradientTracker
				shouldHaveTracker = true
			}
		}
		if !shouldHaveTracker {
			return
		}

		res.GradientTracker = &GradientTracker{
			children: childrenTrackers,
			gradFunc: gradFunc,
		}
	}
}
