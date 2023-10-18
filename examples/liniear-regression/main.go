package main

import (
	"fmt"

	g "github.com/FluffyKebab/gosor"
)

func main() {
	if err := run(); err != nil {
		panic(err)
	}
}

func run() error {
	xs := g.Wrap(g.New(g.WithRange(0, 10, 0.1)))
	ys := xs.Map(func(x float64) float64 {
		return 2*x + 4
	})

	a := g.Wrap(g.New(g.WithValues(-1)))
	b := g.Wrap(g.New(g.WithValues(1)))
	trainingRate := g.Wrap(g.New(g.WithValues(0.1)))

	fmt.Println("training for function: f(x) = 2x + 4")

	for i := 0; i < 100; i++ {
		loss, err := xs.
			Do(g.Mul, a).
			Do(g.Add, b).
			Do(g.Sub, ys).
			DoT(g.Square).
			DoT(g.Sum).
			Do(g.Div, g.Wrap(g.New(g.WithValues(float64(ys.MustValue().Len()))))).
			Value()
		if err != nil {
			return fmt.Errorf("error training: %w", err)
		}

		err = loss.Backward(nil)
		if err != nil {
			return fmt.Errorf("error training: %w", err)
		}

		a.DoInto(a, g.SubInto, a.Gradient().Do(g.Mul, trainingRate))
		b.DoInto(b, g.SubInto, b.Gradient().Do(g.Mul, trainingRate))

		a.MustValue().ResetGradient()
		b.MustValue().ResetGradient()
	}

	fmt.Printf("result: f(x) = %fx + %f", a.MustValue().Item(), b.MustValue().Item())
	return nil
}
