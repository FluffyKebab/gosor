package gosor

type MaybeTensor struct {
	t   *Tensor
	err error
}

func (t *MaybeTensor) Unwrap() (*Tensor, error) {
	return t.t, t.err
}

func Wrap(t *Tensor) *MaybeTensor {
	return &MaybeTensor{t, nil}
}

func WrapErr(err error) *MaybeTensor {
	return &MaybeTensor{nil, err}
}

func unwrapTwo(m1, m2 *MaybeTensor) (*Tensor, *Tensor, error) {
	t1, err := m1.Unwrap()
	if err != nil {
		return nil, nil, err
	}
	t2, err := m2.Unwrap()
	if err != nil {
		return nil, nil, err
	}

	return t1, t2, err
}
