package gosor

import "fmt"

func Add(t1, t2 *Tensor) (*Tensor, error) {
	result, err := New(WithSize(t1.sizes...))
	err = elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 + f2 })
	if err != nil {
		return nil, err
	}
	return result, nil
}

func Sub(t1, t2 *Tensor) (*Tensor, error) {
	result, err := New(WithSize(t1.sizes...))
	err = elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 - f2 })
	if err != nil {
		return nil, err
	}
	return result, nil
}

func Mul(t1, t2 *Tensor) (*Tensor, error) {
	result, err := New(WithSize(t1.sizes...))
	err = elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 * f2 })
	if err != nil {
		return nil, err
	}
	return result, nil
}

func Div(t1, t2 *Tensor) (*Tensor, error) {
	result, err := New(WithSize(t1.sizes...))
	err = elementWiseOperation(t1, t2, result, func(f1, f2 float64) float64 { return f1 / f2 })
	if err != nil {
		return nil, err
	}
	return result, nil
}

func elementWiseOperation(t1, t2, result *Tensor, operation func(float64, float64) float64) error {
	if len(t1.sizes) != len(t2.sizes) || len(t1.sizes) != len(result.sizes) {
		return fmt.Errorf("%w: element wise operation with tensors of different sizes", ErrUndefined)
	}

	length := 1
	for i := 0; i < len(t1.sizes); i++ {
		if t1.sizes[i] != t2.sizes[i] || t1.sizes[i] != result.sizes[i] {
			return fmt.Errorf("%w: element wise operation with tensors of different sizes", ErrUndefined)
		}
		length *= t1.sizes[i]
	}

	for i := 0; i < length; i++ {
		result.storage[result.getStorageIndex(i)] = operation(t1.storage[t1.getStorageIndex(i)], t2.storage[t2.getStorageIndex(i)])
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
