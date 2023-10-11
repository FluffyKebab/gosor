package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTensorGet(t *testing.T) {
	t.Parallel()

	tensor, err := New(WithSize(3, 2))
	require.NoError(t, err)
	tensor.storage[0] = 1
	tensor.storage[1] = 2
	tensor.storage[2] = 3
	tensor.storage[5] = 7
	require.Equal(t, 1., tensor.Get(0, 0))
	require.Equal(t, 2., tensor.Get(0, 1))
	require.Equal(t, 3., tensor.Get(1, 0))
	require.Equal(t, 7., tensor.Get(2, 1))

	tensor, err = New(WithSize(2, 2, 2), WithValues(0, 1, 2, 3, 4, 5, 6, 7))
	require.NoError(t, err)
	require.Equal(t, 0., tensor.Get(0, 0, 0))
	require.Equal(t, 1., tensor.Get(0, 0, 1))
	require.Equal(t, 2., tensor.Get(0, 1, 0))
	require.Equal(t, 3., tensor.Get(0, 1, 1))
	require.Equal(t, 4., tensor.Get(1, 0, 0))
	require.Equal(t, 5., tensor.Get(1, 0, 1))
	require.Equal(t, 6., tensor.Get(1, 1, 0))
	require.Equal(t, 7., tensor.Get(1, 1, 1))
}

func TestIndexWithIndex(t *testing.T) {
	t.Parallel()

	tensor, err := New(WithSize(2, 2), WithValues(0, 1, 2, 3))
	require.NoError(t, err)

	subTensor, err := tensor.Index(Index(0))
	require.NoError(t, err)
	require.Equal(t, []float64{0, 1}, subTensor.Items())

	subTensor, err = tensor.Index(Index(1))
	require.NoError(t, err)
	require.Equal(t, []float64{2, 3}, subTensor.Items())

	tensor, err = New(WithSize(2, 2, 2), WithValues(0, 1, 2, 3, 4, 5, 6, 7))
	require.NoError(t, err)

	subTensor, err = tensor.Index(Index(0))
	require.NoError(t, err)
	require.Equal(t, []float64{0, 1, 2, 3}, subTensor.Items())

	subTensor, err = tensor.Index(Index(1))
	require.NoError(t, err)
	require.Equal(t, []float64{4, 5, 6, 7}, subTensor.Items())

}

func TestIndexWithBetween(t *testing.T) {
	t.Parallel()

	t1, err := New(WithSize(5), WithValues(0, 1, 2, 3, 4))
	require.NoError(t, err)

	t2, err := New(WithSize(4, 2), WithValues(
		0, 1,
		2, 3,
		4, 5,
		6, 7,
	))
	require.NoError(t, err)

	t3, err := New(WithSize(3, 2, 2), WithValues(
		0, 1,
		2, 3,

		4, 5,
		6, 7,

		8, 9,
		10, 11,
	))
	require.NoError(t, err)

	testCases := []struct {
		indexers        []Indexer
		tensor          *Tensor
		expectedItems   []float64
		expectedStrides []int
		expectedSizes   []int
		expectedOffset  int
	}{
		{
			indexers:        []Indexer{Between(0, 3)},
			tensor:          t1,
			expectedItems:   []float64{0, 1, 2},
			expectedStrides: []int{1},
			expectedSizes:   []int{3},
			expectedOffset:  0,
		},
		{
			indexers:        []Indexer{Between(1, 2)},
			tensor:          t2,
			expectedItems:   []float64{2, 3},
			expectedStrides: []int{2, 1},
			expectedSizes:   []int{1, 2},
			expectedOffset:  2,
		},
		{
			indexers:        []Indexer{Between(1, 3)},
			tensor:          t2,
			expectedItems:   []float64{2, 3, 4, 5},
			expectedStrides: []int{2, 1},
			expectedSizes:   []int{2, 2},
			expectedOffset:  2,
		},
		{
			indexers:        []Indexer{Between(0, 2)},
			tensor:          t3,
			expectedItems:   []float64{0, 1, 2, 3, 4, 5, 6, 7},
			expectedStrides: []int{4, 2, 1},
			expectedSizes:   []int{2, 2, 2},
			expectedOffset:  0,
		},
	}

	for _, tc := range testCases {
		subTensor, err := tc.tensor.Index(tc.indexers...)
		require.NoError(t, err)
		require.Equal(t, tc.expectedItems, subTensor.Items())
		require.Equal(t, tc.expectedStrides, subTensor.strides)
		require.Equal(t, tc.expectedSizes, subTensor.sizes)
		require.Equal(t, tc.expectedOffset, subTensor.offset)
	}
}

func TestIndexWithBetweenAndIndex(t *testing.T) {
	t.Parallel()

	t1, err := New(WithSize(3, 2, 2), WithValues(
		0, 1,
		2, 3,

		4, 5,
		6, 7,

		0, 1,
		2, 3,
	))
	require.NoError(t, err)

	t2, err := New(WithSize(3, 3), WithValues(
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	))
	require.NoError(t, err)

	testCases := []struct {
		indexers        []Indexer
		tensor          *Tensor
		expectedItems   []float64
		expectedStrides []int
		expectedSizes   []int
		expectedOffset  int
	}{
		{
			indexers:        []Indexer{Between(0, 2), Index(0)},
			tensor:          t1,
			expectedItems:   []float64{0, 1, 4, 5},
			expectedStrides: []int{4, 1},
			expectedSizes:   []int{2, 2},
			expectedOffset:  0,
		},
		{
			indexers:        []Indexer{Between(0, 2), Index(1)},
			tensor:          t1,
			expectedItems:   []float64{2, 3, 6, 7},
			expectedStrides: []int{4, 1},
			expectedSizes:   []int{2, 2},
			expectedOffset:  2,
		},
		{
			indexers:        []Indexer{Between(1, 3), Index(1)},
			tensor:          t1,
			expectedItems:   []float64{6, 7, 2, 3},
			expectedStrides: []int{4, 1},
			expectedSizes:   []int{2, 2},
			expectedOffset:  6,
		},
		{
			indexers:        []Indexer{All(), Index(0)},
			tensor:          t1,
			expectedItems:   []float64{0, 1, 4, 5, 0, 1},
			expectedStrides: []int{4, 1},
			expectedSizes:   []int{3, 2},
			expectedOffset:  0,
		},
		{
			indexers:        []Indexer{All(), All(), Index(0)},
			tensor:          t1,
			expectedItems:   []float64{0, 2, 4, 6, 0, 2},
			expectedStrides: []int{4, 2},
			expectedSizes:   []int{3, 2},
			expectedOffset:  0,
		},
		{
			indexers:        []Indexer{All(), Index(1)},
			tensor:          t2,
			expectedItems:   []float64{1, 4, 7},
			expectedStrides: []int{3},
			expectedSizes:   []int{3},
			expectedOffset:  1,
		},
	}

	for _, tc := range testCases {
		subTensor, err := tc.tensor.Index(tc.indexers...)
		require.NoError(t, err)
		require.Equal(t, tc.expectedItems, subTensor.Items())
		require.Equal(t, tc.expectedStrides, subTensor.strides)
		require.Equal(t, tc.expectedSizes, subTensor.sizes)
		require.Equal(t, tc.expectedOffset, subTensor.offset)
	}
}
