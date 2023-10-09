package gosor

type Tensor struct {
	strides []int
	sizes   []int
	ofset   int
	storage []float64
}

func (t *Tensor) Index(indexes ...int) *Tensor {
	if len(indexes) != len(t.strides) {
		panic("invalid tensor index")
	}

	storageIndex := t.ofset
	for i := 0; i < len(indexes); i++ {
		storageIndex += indexes[i] * t.strides[i]
	}

	return &Tensor{
		strides: []int{1},
		sizes:   []int{1},
		ofset:   storageIndex,
		storage: t.storage,
	}
}

func (t *Tensor) SetIndex(indexes []int, value float64) {
	if len(indexes) != len(t.strides) {
		panic("invalid tensor index")
	}

	storageIndex := t.ofset
	for i := 0; i < len(indexes); i++ {
		storageIndex += indexes[i] * t.strides[i]
	}
	t.storage[storageIndex] = value
}

func (t *Tensor) Items() []float64 {
	return t.storage[t.ofset:]
}

func (t *Tensor) Item() float64 {
	return t.storage[t.ofset]
}

func Zeros(sizes ...int) *Tensor {
	storageLen := 1
	strides := make([]int, len(sizes))
	for i := 0; i < len(sizes); i++ {
		storageLen *= sizes[i]
		if i == len(sizes)-1 {
			strides[i] = 1
			continue
		}
		strides[i] = sizes[i+1]
	}

	return &Tensor{
		strides: strides,
		sizes:   sizes,
		ofset:   0,
		storage: make([]float64, storageLen),
	}
}
