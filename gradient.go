package gosor

import "fmt"

func addAddGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, t1, t2, func() (err error) {
		if res.gradient == nil {
			return fmt.Errorf("gradient for node in front not calculated")
		}

		if t1.gradient == nil {
			t1.gradient, err = New(WithSize(res.gradient.sizes...))
			if err != nil {
				return err
			}
		}
		t1.gradient, err = AddInto(t1.gradient, t1.gradient, res.gradient)
		if err != nil {
			return err
		}

		if t2.gradient == nil {
			t2.gradient, err = New(WithSize(res.gradient.sizes...))
			if err != nil {
				return err
			}
		}
		t2.gradient, err = AddInto(t2.gradient, t2.gradient, res.gradient)
		if err != nil {
			return err
		}

		return nil
	})
}

func addSubGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, t1, t2, func() (err error) {
		if res.gradient == nil {
			return fmt.Errorf("gradient for node in front not calculated")
		}

		if t1.gradient == nil {
			t1.gradient, err = New(WithSize(res.gradient.sizes...))
			if err != nil {
				return err
			}
		}
		t1.gradient, err = AddInto(t1.gradient, t1.gradient, res.gradient)
		if err != nil {
			return err
		}

		if t2.gradient == nil {
			t2.gradient, err = New(WithSize(res.gradient.sizes...))
			if err != nil {
				return err
			}
		}
		t2.gradient, err = SubInto(t2.gradient, t2.gradient, res.gradient)
		if err != nil {
			return err
		}

		return nil
	})
}

func addPowGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, t1, t2, func() (err error) {
		if res.gradient == nil {
			return fmt.Errorf("gradient for node in front not calculated")
		}

		// Todo: calculate gradient for t2.
		if t1.gradient == nil {
			t1.gradient, err = New(WithSize(res.gradient.sizes...))
			if err != nil {
				return err
			}
		}

		// t1.grad += (t2 * t1**(t2-1)) * res.grad
		tensor1 := Wrap(t1, nil)
		tensor2 := Wrap(t2, nil)
		resultGrad := Wrap(res.gradient, nil)

		curGradient := resultGrad.Do(Mul, tensor2.Do(
			Mul,
			tensor1.Do(
				Pow,
				tensor2.Do(Sub, Wrap(New(WithValues(1)))),
			),
		))

		t1.gradient, err = Wrap(t1.gradient, nil).DoInto(Wrap(t1.gradient, nil), AddInto, curGradient).Value()
		if err != nil {
			return err
		}

		return nil
	})
}
