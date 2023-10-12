package gosor

import "fmt"

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

func addAddGradientTrackerToRes(res, t1, t2 *Tensor) {
	if t1.GradientTracker != nil || t2.GradientTracker != nil {
		res.GradientTracker = &GradientTracker{
			children: []*GradientTracker{t1.GradientTracker, t2.GradientTracker},
			gradFunc: func() (err error) {
				if res.gradient == nil {
					return fmt.Errorf("gradient for node in front not calculated")
				}

				if t1.gradient == nil {
					t1.gradient, err = res.gradient.DeepCopy()
					if err != nil {
						return err
					}
				} else {
					t1.gradient, err = AddInto(t1.gradient, t1.gradient, res.gradient)
					if err != nil {
						return err
					}
				}

				if t2.gradient == nil {
					t2.gradient, err = res.gradient.DeepCopy()
					if err != nil {
						return err
					}
				} else {
					t2.gradient, err = AddInto(t2.gradient, t2.gradient, res.gradient)
					if err != nil {
						return err
					}
				}

				return nil
			},
		}
	}
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

	addAddGradientTrackerToRes(result, t1, t2)
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

func elementWiseOperationInto(
	result,
	t1,
	t2 *Tensor,
	operation func(float64, float64) float64,
) (t *Tensor, err error) {
	if len(t1.sizes) == 0 {
		return nil, ErrInvalidTensor
	}
	if result == nil {
		result, err = New(WithSize(t1.sizes...))
		if err != nil {
			return nil, err
		}
	}
	if len(t1.sizes) != len(t1.strides) {
		return nil, ErrInvalidTensor
	}
	if len(t2.sizes) != len(t2.strides) {
		return nil, ErrInvalidTensor
	}
	if len(t2.sizes) != len(t2.strides) {
		return nil, ErrInvalidTensor
	}
	if len(t1.sizes) != len(t2.sizes) || len(t1.sizes) != len(result.sizes) {
		return nil, fmt.Errorf("%w: element wise operation with tensors of different dimensions", ErrUndefined)
	}

	length := 1
	for i := 0; i < len(t1.sizes); i++ {
		if t1.sizes[i] != t2.sizes[i] || t1.sizes[i] != result.sizes[i] {
			return nil, fmt.Errorf("%w: element wise operation with tensors of different sizes", ErrUndefined)
		}
		length *= t1.sizes[i]
	}

	for i := 0; i < length; i++ {
		result.storage[result.getStorageIndex(i)] = operation(t1.storage[t1.getStorageIndex(i)], t2.storage[t2.getStorageIndex(i)])
	}

	return result, nil
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
