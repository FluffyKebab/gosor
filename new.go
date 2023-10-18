package gosor

import (
	"fmt"
)

type newOptions struct {
	size           []int
	storage        []float64
	isNotLeaf      bool
	storageCreator func() ([]float64, error)
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

func WithRange(start, end, step float64) newOption {
	return func(no *newOptions) {
		no.storageCreator = func() ([]float64, error) {
			i := start
			res := make([]float64, 0, int((end-start)/step))
			for i < end {
				res = append(res, i)
				i += step
			}
			return res, nil
		}
	}
}

func withIsNotLeaf() newOption {
	return func(no *newOptions) {
		no.isNotLeaf = true
	}
}

func New(opts ...newOption) (t *Tensor, err error) {
	var options newOptions
	for _, opt := range opts {
		opt(&options)
	}

	if options.storageCreator != nil {
		options.storage, err = options.storageCreator()
		if err != nil {
			return nil, fmt.Errorf("%w: %w", ErrInvalidTensorCreation, err)
		}
	}

	if options.size == nil && options.storage == nil {
		return nil, fmt.Errorf("%w: must specify at least one of size or storage values", ErrInvalidTensorCreation)
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
		isNotLeaf:       options.isNotLeaf,
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
