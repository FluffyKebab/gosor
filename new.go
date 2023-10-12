package gosor

import (
	"fmt"
)

type newOptions struct {
	size    []int
	storage []float64
}

type newOption func(*newOptions)

func WithSize(size ...int) newOption {
	return func(no *newOptions) {
		no.size = size
	}
}

func WithValues(s ...float64) newOption {
	return func(no *newOptions) {
		no.storage = s
	}
}

func New(opts ...newOption) (*Tensor, error) {
	var options newOptions
	for _, opt := range opts {
		opt(&options)
	}

	if options.size == nil && options.storage == nil {
		return nil, fmt.Errorf("%w: must have either/both of size or storage", ErrInvalidTensorCreation)
	}

	if options.size == nil {
		options.size = []int{len(options.storage)}
	}

	storageLen, strides := getStorageLenAndStridesFromSize(options.size)
	if options.storage == nil {
		options.storage = make([]float64, storageLen)
	}

	if storageLen != len(options.storage) {
		return nil, fmt.Errorf("wrong length of storage for sizes. Got: %d. Want: %d", len(options.storage), storageLen)
	}

	return &Tensor{
		GradientTracker: &GradientTracker{},
		strides:         strides,
		sizes:           options.size,
		offset:          0,
		storage:         options.storage,
	}, nil
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
