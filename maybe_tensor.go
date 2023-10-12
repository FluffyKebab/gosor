package gosor

type MaybeTensor struct {
	t   *Tensor
	err error
}

func (m1 *MaybeTensor) Do(f func(*Tensor, *Tensor) (*Tensor, error), m2 *MaybeTensor) *MaybeTensor {
	if m1.err != nil {
		return m1
	}
	if m2.err != nil {
		return m2
	}

	result, err := f(m1.t, m2.t)
	return &MaybeTensor{t: result, err: err}
}

func (m *MaybeTensor) DoT(f func(*Tensor) (*Tensor, error)) *MaybeTensor {
	if m.err != nil {
		return m
	}

	result, err := f(m.t)
	return &MaybeTensor{t: result, err: err}
}

func (m *MaybeTensor) Index(indexes ...Indexer) *MaybeTensor {
	if m.err != nil {
		return m
	}

	result, err := m.t.Index(indexes...)
	return &MaybeTensor{t: result, err: err}
}

func (t *MaybeTensor) Value() (*Tensor, error) {
	if t.err != nil {
		return nil, t.err
	}
	return t.t, nil
}

func (t *MaybeTensor) MustValue() *Tensor {
	if t.err != nil {
		panic(t.err.Error())
	}
	return t.t
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
