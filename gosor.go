package gosor

import (
	"fmt"
)

type Tensor struct {
	strides []int
	sizes   []int
	ofset   int
	storage []float64
}

func New(sizes []int, storage []float64) (*Tensor, error) {
	storageLen, strides := getStorageLenAndStridesFromSize(sizes)
	if storageLen != len(storage) {
		return nil, fmt.Errorf("wrong length of storage for sizes. Got: %d. Want: %d", len(storage), storageLen)
	}
	return &Tensor{
		strides: strides,
		sizes:   sizes,
		ofset:   0,
		storage: storage,
	}, nil
}

func NewZeros(sizes ...int) *Tensor {
	storageLen, strides := getStorageLenAndStridesFromSize(sizes)
	return &Tensor{
		strides: strides,
		sizes:   sizes,
		ofset:   0,
		storage: make([]float64, storageLen),
	}
}

func getStorageLenAndStridesFromSize(sizes []int) (int, []int) {
	storageLen := 1
	strides := make([]int, len(sizes))
	for i := len(sizes) - 1; i >= 0; i-- {
		storageLen *= sizes[i]
		if i == len(sizes)-1 {
			strides[i] = 1
			continue
		}
		strides[i] = sizes[i+1] * strides[i+1]
	}

	return storageLen, strides
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
		index := t.ofset
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
	return t.storage[t.ofset]
}

func (t *Tensor) ShallowCopy() *Tensor {
	return &Tensor{strides: t.strides, sizes: t.sizes, ofset: t.ofset, storage: t.storage}
}
