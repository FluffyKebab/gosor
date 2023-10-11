package gosor

type MaybeTensor struct {
	t   *Tensor
	err error
}

func (m *MaybeTensor) Do(f func(*Tensor, *Tensor) (*Tensor, error), t *Tensor) *MaybeTensor {
	if m.err != nil {
		return m
	}

	result, err := f(m.t, t)
	return &MaybeTensor{t: result, err: err}
}

func (m *MaybeTensor) DoT(f func(*Tensor) (*Tensor, error)) *MaybeTensor {
	if m.err != nil {
		return m
	}

	result, err := f(m.t)
	return &MaybeTensor{t: result, err: err}
}

func (t *MaybeTensor) Value() (*Tensor, error) {
	if t.err != nil {
		return nil, t.err
	}
	return t.t, nil
}

func Wrap(t *Tensor, err error) *MaybeTensor {
	return &MaybeTensor{
		t:   t,
		err: err,
	}
}

/*

for i := 0; i < len(weights); i++ {
	outputs, err := inputs.Do(gosor.Dot, wights[i]).Do(gosor.Add, bias[i]).DoT(gosor.Tanh).Value()
	if err != nil {
		return err
	}
	inputs = outputs
}

loss, err := expected.Do(gsor.Sub, inputs).Do(gsor.Square).DoT(gsor.Sum).Value()
if err != nil {
	return err
}

loss.backwards()
*/
