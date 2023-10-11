package gosor

type Tensor struct {
	storage []float64
	strides []int
	sizes   []int
	offset  int
}

func (t *Tensor) Items() []float64 {
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
