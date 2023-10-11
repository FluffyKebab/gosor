package gosor

import "fmt"

func AddW(mt1, mt2 *MaybeTensor) *MaybeTensor {
	t1, t2, err := unwrapTwo(mt1, mt2)
	if err != nil {
		return WrapErr(err)
	}

	return Add(t1, t2)
}
func Add(t1, t2 *Tensor) *MaybeTensor {
	result := NewZeros(t1.sizes...)
	err := elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 + f2 })
	if err != nil {
		return WrapErr(err)
	}
	return Wrap(result)
}
func (mt1 *MaybeTensor) Add(mt2 *MaybeTensor) *MaybeTensor {
	t1, t2, err := unwrapTwo(mt1, mt2)
	if err != nil {
		mt1.err = err
		return mt1
	}
	err = t1.Add(t2)
	mt1.err = err
	return mt1
}
func (t1 *Tensor) Add(t2 *Tensor) error {
	err := elementWiseOperation(t1, t2, t1, func(f1, f2 float64) float64 { return f1 + f2 })
	if err != nil {
		return err
	}
	return nil
}

func MulW(mt1, mt2 *MaybeTensor) *MaybeTensor {
	t1, t2, err := unwrapTwo(mt1, mt2)
	if err != nil {
		return WrapErr(err)
	}

	return Add(t1, t2)
}
func Mul(t1, t2 *Tensor) *MaybeTensor {
	result := NewZeros(t1.sizes...)
	err := elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 * f2 })
	if err != nil {
		return WrapErr(err)
	}
	return Wrap(result)
}
func (mt1 *MaybeTensor) Mul(mt2 *MaybeTensor) *MaybeTensor {
	t1, t2, err := unwrapTwo(mt1, mt2)
	if err != nil {
		mt1.err = err
		return mt1
	}
	err = t1.Add(t2)
	mt1.err = err
	return mt1
}
func (t1 *Tensor) Mul(t2 *Tensor) error {
	err := elementWiseOperation(t1, t2, t1, func(f1, f2 float64) float64 { return f1 * f2 })
	if err != nil {
		return err
	}
	return nil
}

func SubW(mt1, mt2 *MaybeTensor) *MaybeTensor {
	t1, t2, err := unwrapTwo(mt1, mt2)
	if err != nil {
		return WrapErr(err)
	}

	return Add(t1, t2)
}
func Sub(t1, t2 *Tensor) *MaybeTensor {
	result := NewZeros(t1.sizes...)
	err := elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 - f2 })
	if err != nil {
		return WrapErr(err)
	}
	return Wrap(result)
}
func (mt1 *MaybeTensor) Sub(mt2 *MaybeTensor) *MaybeTensor {
	t1, t2, err := unwrapTwo(mt1, mt2)
	if err != nil {
		mt1.err = err
		return mt1
	}
	err = t1.Add(t2)
	mt1.err = err
	return mt1
}
func (t1 *Tensor) Sub(t2 *Tensor) error {
	err := elementWiseOperation(t1, t2, t1, func(f1, f2 float64) float64 { return f1 - f2 })
	if err != nil {
		return err
	}
	return nil
}

func elementWiseOperation(t1, t2, result *Tensor, operation func(float64, float64) float64) error {
	if len(t1.sizes) != len(t2.sizes) || len(t1.sizes) != len(result.sizes) {
		return fmt.Errorf("%w: element wise addition with tensors of different sizes", ErrUndefined)
	}

	length := 1
	for i := 0; i < len(t1.sizes); i++ {
		if t1.sizes[i] != t2.sizes[i] || t1.sizes[i] != result.sizes[i] {
			return fmt.Errorf("%w: element wise addition with tensors of different sizes", ErrUndefined)
		}
		length *= t1.sizes[i]
	}

	for i := 0; i < length; i++ {
		result.storage[result.getStorageIndex(i)] = t1.storage[t1.getStorageIndex(i)] + t2.storage[t2.getStorageIndex(i)]
	}

	return nil
}

func (t *Tensor) getStorageIndex(i int) int {
	index := t.offset
	v := i
	for j := len(t.sizes) - 1; j >= 0; j-- {
		dimensionIndex := v % t.sizes[j]
		v /= t.sizes[j]
		index += dimensionIndex * t.strides[j]
	}
	return index
}
