package main

import (
	"fmt"

	g "github.com/FluffyKebab/gosor"
)

func main() {
	if err := run(); err != nil {
		panic(err.Error())
	}
}

func run() error {
	input := g.Wrap(g.New(g.WithValues(5)))
	bias := g.Wrap(g.New(g.WithValues(2)))
	wantedOutput := g.Wrap(g.New(g.WithValues(1)))
	learningRate := g.Wrap(g.New(g.WithValues(0.1)))

	for i := 0; i < 100; i++ {
		output := input.Do(g.Add, bias)
		modelError := wantedOutput.Do(g.Sub, output)
		loss, err := modelError.DoT(g.Square).Value()
		if err != nil {
			return fmt.Errorf("training failed: %w", err)
		}

		fmt.Println("loss: ", loss)

		err = loss.Backward(nil)
		if err != nil {
			return fmt.Errorf("training failed: %w", err)
		}

		biasGradient := g.Wrap(bias.MustValue().Gradient())
		bias.DoInto(bias, g.SubInto, biasGradient.Do(g.Mul, learningRate))
		bias.MustValue().ResetGradient()

	}

	fmt.Println("Input: ", input)
	fmt.Println("Bias: ", bias)
	fmt.Println("Wanted output: ", wantedOutput)
	fmt.Println("Actual output: ", input.Do(g.Add, bias))

	return nil
}
