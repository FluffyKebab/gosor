package gosor

import "fmt"

type GradFunc func() ([]*Tensor, error)

type GradientTracker struct {
	children []*GradientTracker
	gradient *Tensor
	gradFunc GradFunc
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
	if previousGradient == nil {
		previousGradient = Wrap(New(WithValues(1))).MustValue()
	}
	fmt.Println("doing backward with prev grad: ", previousGradient)

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
		return nil
	}
	localGradients, err := g.gradFunc()
	if err != nil {
		return fmt.Errorf("calculating local gradient: %w", err)
	}
	if len(localGradients) != len(g.children) {
		return fmt.Errorf("wrong amount tensors returned from grad func")
	}

	globalGradient := localGradients
	for _, gradient := range globalGradient {
		if gradient == nil {
			continue
		}
		_, err := MulInto(gradient, gradient, previousGradient)
		if err != nil {
			return err
		}
	}

	for i := 0; i < len(localGradients); i++ {
		if globalGradient[i] != nil {
			g.children[i].Backward(globalGradient[i])
		}
	}

	return nil
}

func addGradientTracker(res, t1, t2 *Tensor, gradFunc GradFunc) {
	if res.isNotLeaf && (res.gradFunc == nil && (t1.GradientTracker != nil || t2.GradientTracker != nil)) {
		res.GradientTracker = &GradientTracker{
			children: []*GradientTracker{t1.GradientTracker, t2.GradientTracker},
			gradFunc: gradFunc,
		}
	}
}
