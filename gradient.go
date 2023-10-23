package gosor

func addAddGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, []*Tensor{t1, t2}, func() ([]*Tensor, error) {
		return []*Tensor{
			Wrap(New(WithValues(1))).MustValue(),
			Wrap(New(WithValues(1))).MustValue(),
		}, nil
	})
}

func addSubGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, []*Tensor{t1, t2}, func() ([]*Tensor, error) {
		return []*Tensor{
			Wrap(New(WithValues(1))).MustValue(),
			Wrap(New(WithValues(-1))).MustValue(),
		}, nil
	})
}

func addMulGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, []*Tensor{t1, t2}, func() ([]*Tensor, error) {
		t1LocalGrad, err := t2.DeepCopy()
		if err != nil {
			return nil, err
		}
		t2LocalGrad, err := t1.DeepCopy()
		if err != nil {
			return nil, err
		}
		return []*Tensor{
			t1LocalGrad,
			t2LocalGrad,
		}, nil
	})
}

func addPowGradientTrackerToRes(res, t1, t2 *Tensor) {
	addGradientTracker(res, []*Tensor{t1, t2}, func() ([]*Tensor, error) {
		// localGradient = t2 * t1**(t2-1)
		tensor1 := Wrap(t1, nil)
		tensor2 := Wrap(t2, nil)
		resultGrad := Wrap(res.gradient, nil)

		t1Grad, err := resultGrad.Do(Mul, tensor2.Do(
			Mul,
			tensor1.Do(
				Pow,
				tensor2.Do(Sub, Wrap(New(WithValues(1)))),
			),
		)).Value()
		if err != nil {
			return nil, err
		}

		// Todo: calculate gradient for t2.

		return []*Tensor{t1Grad, nil}, nil
	})
}
