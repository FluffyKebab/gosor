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
	_ = ys

	a := g.Wrap(g.New(g.WithValues(-1)))
	b := g.Wrap(g.New(g.WithValues(1)))
	trainingRate := g.Wrap(g.New(g.WithValues(0.1)))

	fmt.Println("training for function: 2x + 4")

	for i := 0; i < 100; i++ {
		loss, err := xs.Do(g.Mul, a).Do(g.Add, b). /* .Do(g.MSE, ys) */ Value()
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

	fmt.Println()
	fmt.Println("result loss: ", xs.Do(g.Mul, a).Do(g.Add, b))
	return nil
}
