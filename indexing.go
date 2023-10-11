package gosor

import (
	"fmt"
)

type Indexer interface {
	IndexIn(t *Tensor, dim int) (int, error)
}

type index struct {
	pos int
}

var _ Indexer = index{}

func Index(i int) index {
	return index{i}
}

func (i index) IndexIn(t *Tensor, dim int) (int, error) {
	if len(t.sizes) == 0 {
		return 0, fmt.Errorf("%w: no shape", ErrInvalidTensor)
	}
	if len(t.sizes) != len(t.strides) {
		return 0, fmt.Errorf("%w: length of strides do not match length of sizes", ErrInvalidTensor)
	}
	if dim >= len(t.sizes) {
		return 0, ErrIndexOutOfBounds
	}
	if i.pos >= t.sizes[dim] {
		return 0, ErrIndexOutOfBounds
	}

	t.offset += t.strides[dim] * i.pos

	if len(t.sizes) == 1 {
		t.sizes = []int{1}
		t.strides = []int{1}
		return 0, nil
	}

	newSize := make([]int, 0, len(t.sizes)-1)
	newStride := make([]int, 0, len(t.strides)-1)
	for i := 0; i < len(t.sizes); i++ {
		if i == dim {
			continue
		}
		newSize = append(newSize, t.sizes[i])
		newStride = append(newStride, t.strides[i])
	}

	t.sizes = newSize
	t.strides = newStride
	return 0, nil
}

type between struct {
	start int
	end   int
}

var _ Indexer = between{}

func Between(start int, end int) between {
	return between{start, end}
}

func All() between {
	return between{0, -1}
}

func (b between) IndexIn(t *Tensor, dim int) (int, error) {
	if len(t.strides) == 0 {
		return 0, ErrInvalidTensor
	}
	if len(t.sizes) == 0 {
		return 0, ErrInvalidTensor
	}
	if b.end > t.sizes[0] {
		return 0, fmt.Errorf("%w: end value in between", ErrIndexOutOfBounds)
	}

	t.offset += b.start * t.strides[0]

	if b.start > b.end {
		if b.end == -1 {
			return dim + 1, nil
		}
		return 0, ErrIndexOutOfBounds
	}

	t.sizes[0] = b.end - b.start
	return dim + 1, nil
}

func (t *Tensor) Index(indexes ...Indexer) *MaybeTensor {
	var err error

	tensor := t.ShallowCopy()
	dim := 0
	for _, indexer := range indexes {
		dim, err = indexer.IndexIn(tensor, dim)
		if err != nil {
			return WrapErr(err)
		}
	}

	return Wrap(tensor)
}

func (t *Tensor) Get(indexes ...int) float64 {
	if len(indexes) != len(t.strides) {
		panic("get invalid tensor index")
	}

	storageIndex := t.offset
	for i := 0; i < len(indexes); i++ {
		storageIndex += indexes[i] * t.strides[i]
	}

	return t.storage[storageIndex]
}

func (t *Tensor) Set(indexes []int, value float64) {
	if len(indexes) != len(t.strides) {
		panic("set invalid tensor index")
	}

	storageIndex := t.offset
	for i := 0; i < len(indexes); i++ {
		storageIndex += indexes[i] * t.strides[i]
	}
	t.storage[storageIndex] = value
}
