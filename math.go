package gosor

import (
	"fmt"
	"math"
)

// Add allocates memory for a result tensor and does element wise addition
// into the result tensor using t1 and t2.
func Add(t1, t2 *Tensor) (*Tensor, error) {
	return AddInto(nil, t1, t2)
}

// AddInto does element wise addition into the result tensor. If the result is
// nil, a new tensor is allocated.
func AddInto(result, t1, t2 *Tensor) (*Tensor, error) {
	result, err := elementWiseOperationInto(result, t1, t2, func(f1, f2 float64) float64 { return f1 + f2 })
	if err != nil {
		return nil, err
	}

	addAddGradientTrackerToRes(result, t1, t2)
	return result, nil
}

// Sub allocates memory for a result tensor and does element wise subtraction
// into the result tensor using t1 and t2.
func Sub(t1, t2 *Tensor) (*Tensor, error) {
	return SubInto(nil, t1, t2)
}

// SubInto does element wise subtraction into the result tensor. If the result is
// nil, a new tensor is allocated.
func SubInto(result, t1, t2 *Tensor) (*Tensor, error) {
	result, err := elementWiseOperationInto(result, t1, t2, func(f1, f2 float64) float64 { return f1 - f2 })
	if err != nil {
		return nil, err
	}

	addSubGradientTrackerToRes(result, t1, t2)
	return result, nil
}

// Mul allocates memory for a result tensor and does element wise multiplication
// into the result tensor using t1 and t2.
func Mul(t1, t2 *Tensor) (*Tensor, error) {
	return MulInto(nil, t1, t2)
}

// MulInto does element wise multiplication into the result tensor. If the
// result is nil, a new tensor is allocated.
func MulInto(result, t1, t2 *Tensor) (*Tensor, error) {
	result, err := elementWiseOperationInto(result, t1, t2, func(f1, f2 float64) float64 { return f1 * f2 })
	if err != nil {
		return nil, err
	}

	addAddGradientTrackerToRes(result, t1, t2)
	return result, nil
}

// Div allocates memory for a result tensor and does element wise division
// into the result tensor using t1 and t2.
func Div(t1, t2 *Tensor) (*Tensor, error) {
	return MulInto(nil, t1, t2)
}

// DivInto does element wise division into the result tensor. If the
// result is nil, a new tensor is allocated.
func DivInto(result, t1, t2 *Tensor) (*Tensor, error) {
	result, err := elementWiseOperationInto(result, t1, t2, func(f1, f2 float64) float64 { return f1 / f2 })
	if err != nil {
		return nil, err
	}

	addAddGradientTrackerToRes(result, t1, t2)
	return result, nil
}

// Div allocates memory for a result tensor and does element wise division
// into the result tensor using t1 and t2.
func Pow(t1, t2 *Tensor) (*Tensor, error) {
	return PowInto(nil, t1, t2)
}

// DivInto does element wise division into the result tensor. If the
// result is nil, a new tensor is allocated.
func PowInto(result, t1, t2 *Tensor) (*Tensor, error) {
	result, err := elementWiseOperationInto(result, t1, t2, func(f1, f2 float64) float64 { return math.Pow(f1, f2) })
	if err != nil {
		return nil, err
	}

	addPowGradientTrackerToRes(result, t1, t2)
	return result, nil
}

func Square(t *Tensor) (*Tensor, error) {
	return SquareInto(nil, t)
}

func SquareInto(result, t1 *Tensor) (*Tensor, error) {
	two, _ := New(WithValues(2))
	return PowInto(result, t1, two)
}

func elementWiseOperationInto(
	result,
	t1,
	t2 *Tensor,
	operation func(float64, float64) float64,
) (t *Tensor, err error) {
	if len(t1.sizes) == 0 || len(t2.sizes) == 0 {
		return nil, ErrInvalidTensor
	}

	t1, t2, err = Broadcast(t1, t2)
	if err != nil {
		return nil, err
	}

	if result == nil {
		result, err = New(WithSize(t1.sizes...), withIsNotLeaf())
		if err != nil {
			return nil, err
		}
	} else {
		if len(t1.sizes) != len(result.sizes) {
			return nil, fmt.Errorf("wrong result size")
		}
		for i := 0; i < len(t1.sizes); i++ {
			if t1.sizes[i] != result.sizes[i] {
				return nil, fmt.Errorf("wrong result size")
			}
		}
	}

	length := 1
	for i := 0; i < len(t1.sizes); i++ {
		length *= t1.sizes[i]
	}

	for i := 0; i < length; i++ {
		result.storage[result.getStorageIndex(i)] = operation(t1.storage[t1.getStorageIndex(i)], t2.storage[t2.getStorageIndex(i)])
	}

	return result, nil
}

// Broadcast broadcast two tensors to be compatible for element wise
// operations. Broadcasting follows two rules:
//   - If the tensors have different ranks, the smaller tensor is padded with
//     ones on its left until both tensors have the same rank.
//   - If the corresponding dimensions of the two tensors have the same size,
//     or one of them is one, these tensors are compatible.
func Broadcast(a, b *Tensor) (*Tensor, *Tensor, error) {
	a = a.ShallowCopy()
	b = b.ShallowCopy()

	a.sizes, b.sizes = makeSameSize(a.sizes, b.sizes, 1)
	a.strides, b.strides = makeSameSize(a.strides, b.strides, 0)

	for i := 0; i < len(a.sizes); i++ {
		if a.sizes[i] == b.sizes[i] {
			continue
		}

		if a.sizes[i] == 1 {
			a.strides[i] = 0
			a.sizes[i] = b.sizes[i]
			continue
		}
		if b.sizes[i] == 1 {
			b.strides[i] = 0
			b.sizes[i] = a.sizes[i]
			continue
		}

		return nil, nil, ErrFieldsMismatch
	}

	return a, b, nil
}

func makeSameSize(a, b []int, filler int) ([]int, []int) {
	if len(a) > len(b) {
		b, a = makeSameSize(b, a, filler)
		return a, b
	}
	if len(a) < len(b) {
		numToAdd := len(b) - len(a)
		numLeftToAdd := numToAdd
		newA := make([]int, len(b))

		for i := 0; i < len(newA); i++ {
			if numLeftToAdd > 0 {
				newA[i] = filler
				numLeftToAdd--
				continue
			}
			newA[i] = a[i-numToAdd]
		}
		return newA, b
	}

	return a, b
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
