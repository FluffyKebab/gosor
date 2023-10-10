package gosor

import (
	"fmt"
)

type Indexer interface {
	IndexIn(t *Tensor) error
}

type index struct {
	pos int
}

var _ Indexer = index{}

func Index(i int) index {
	return index{i}
}

func (i index) IndexIn(t *Tensor) error {
	if len(t.sizes) == 0 {
		return fmt.Errorf("%w: missing shape", ErrInvalidTensor)
	}
	if i.pos >= t.sizes[0] {
		return ErrIndexOutOfBounds
	}

	t.ofset = t.strides[0] * i.pos

	if len(t.sizes) == 1 {
		t.sizes = []int{1}
		t.strides = []int{1}
		return nil
	}

	t.sizes = t.sizes[1:]
	t.strides = t.strides[1:]
	return nil
}

type between struct {
	start int
	end   int
}

func Between(start int, end int) between {
	return between{start, end}
}

var All = between{start: -1, end: -1}

func (t *Tensor) Index(indexes ...Indexer) (*Tensor, error) {
	t = t.ShallowCopy()
	for _, indexer := range indexes {
		err := indexer.IndexIn(t)
		if err != nil {
			return nil, err
		}
	}
	return t, nil
}

func (t *Tensor) Get(indexes ...int) float64 {
	if len(indexes) != len(t.strides) {
		panic("get invalid tensor index")
	}

	storageIndex := t.ofset
	for i := 0; i < len(indexes); i++ {
		storageIndex += indexes[i] * t.strides[i]
	}

	return t.storage[storageIndex]
}

func (t *Tensor) Set(indexes []int, value float64) {
	if len(indexes) != len(t.strides) {
		panic("set invalid tensor index")
	}

	storageIndex := t.ofset
	for i := 0; i < len(indexes); i++ {
		storageIndex += indexes[i] * t.strides[i]
	}
	t.storage[storageIndex] = value
}
