package gosor

import "fmt"

func New(sizes []int, storage []float64) (*Tensor, error) {
	storageLen, strides := getStorageLenAndStridesFromSize(sizes)
	if storageLen != len(storage) {
		return nil, fmt.Errorf("wrong length of storage for sizes. Got: %d. Want: %d", len(storage), storageLen)
	}
	return &Tensor{
		strides: strides,
		sizes:   sizes,
		offset:  0,
		storage: storage,
	}, nil
}

func NewZeros(sizes ...int) *Tensor {
	storageLen, strides := getStorageLenAndStridesFromSize(sizes)
	return &Tensor{
		strides: strides,
		sizes:   sizes,
		offset:  0,
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
