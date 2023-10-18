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

	return Wrap(f(m1.t, m2.t))
}

func (m1 *MaybeTensor) DoInto(
	result *MaybeTensor,
	f func(*Tensor, *Tensor, *Tensor) (*Tensor, error),
	m2 *MaybeTensor,
) *MaybeTensor {
	if m1.err != nil {
		return m1
	}
	if m2.err != nil {
		return nil
	}

	return Wrap(f(result.t, m1.t, m2.t))
}

func (m *MaybeTensor) DoT(f func(*Tensor) (*Tensor, error)) *MaybeTensor {
	if m.err != nil {
		return m
	}

	return Wrap(f(m.t))
}

func (m *MaybeTensor) Index(indexes ...Indexer) *MaybeTensor {
	if m.err != nil {
		return m
	}

	return Wrap(m.t.Index(indexes...))
}

func (m *MaybeTensor) Gradient() *MaybeTensor {
	if m.err != nil {
		return m
	}

	return Wrap(m.t.Gradient())
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

func (t *MaybeTensor) String() string {
	if t.err != nil {
		return t.err.Error()
	}
	return t.t.String()
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
