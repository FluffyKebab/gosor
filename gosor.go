package gosor

import "fmt"

type Tensor struct {
	*GradientTracker
	storage   []float64
	strides   []int
	sizes     []int
	offset    int
	isNotLeaf bool
}

func (t *Tensor) Items() []float64 {
	if t == nil {
		panic("items on nil tensor")
	}
	if len(t.sizes) != len(t.strides) {
		panic("items on invalid tensor")
	}

	size := 1
	for _, i := range t.sizes {
		size *= i
	}
	items := make([]float64, size)

	for i := 0; i < size; i++ {
		index := t.offset
		v := i
		for j := len(t.sizes) - 1; j >= 0; j-- {
			dimensionIndex := v % t.sizes[j]
			v /= t.sizes[j]
			index += dimensionIndex * t.strides[j]
		}

		items[i] = t.storage[index]
	}

	return items
}

func (t *Tensor) Item() float64 {
	return t.storage[t.offset]
}

// ShallowCopy copies everything except the underling storage of the tensor.
func (t *Tensor) ShallowCopy() *Tensor {
	strides := make([]int, len(t.strides))
	copy(strides, t.strides)

	sizes := make([]int, len(t.sizes))
	copy(sizes, t.sizes)

	return &Tensor{strides: strides, sizes: sizes, offset: t.offset, storage: t.storage}
}

func (t *Tensor) DeepCopy() (*Tensor, error) {
	return New(WithSize(t.sizes...), WithValues(t.Items()...))
}

func (t *Tensor) String() string {
	r := "tensor["
	for i := 0; i < len(t.sizes); i++ {
		r += fmt.Sprint(t.sizes[i])
		if i+1 != len(t.sizes) {
			r += "*"
		}
	}
	r += "]{ "

	for _, item := range t.Items() {
		r += fmt.Sprint(item, " ")
	}

	return r + "}"
}

func Map(t *Tensor, f func(f float64) float64) (*Tensor, error) {
	size := 1
	for _, i := range t.sizes {
		size *= i
	}
	storage := make([]float64, size)

	for i := 0; i < len(storage); i++ {
		storage[i] = f(t.storage[t.getStorageIndex(i)])
	}

	return New(withIsNotLeaf(), WithSize(t.sizes...), WithValues(storage...))
}
