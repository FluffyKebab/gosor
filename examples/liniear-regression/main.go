package main

/* func main() {
	if err := run(); err != nil {
		panic(err)
	}
}

func run() error {
	xs, err := g.New(g.WithRange(0, 10, 0.1))
	if err != nil {
		return err
	}
	ys := g.Wrap(g.New(
		g.WithBacking(func(x float64) float64 { return x*2.0 + 4 }),
		g.WithSize(xs.LenItems()),
	))

	a := g.Wrap(g.New(g.WithValues(-1)))
	b := g.Wrap(g.New(g.WithValues(1)))
	trainingRate := g.Wrap(g.New(g.WithValues(0.1)))

	fmt.Println("training for function: 2x + 4")

	for i := 0; i < 100; i++ {
		loss, err := g.Wrap(xs, nil).Do(g.Mul, a).Do(g.Add, b).Do(g.MSE, ys).Value()
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

	fmt.Println("result loss: ", g.Wrap(xs, nil).Do(g.Mul, a).Do(g.Add, b).Do(g.MSE, ys))
	fmt.Println("a: ", a)
	fmt.Println("b: ", b)
	return nil
} */
