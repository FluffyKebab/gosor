package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTensorGet(t *testing.T) {
	t.Parallel()

	tensor := NewZeros(3, 2)
	tensor.storage[0] = 1
	tensor.storage[1] = 2
	tensor.storage[2] = 3
	tensor.storage[5] = 7
	require.Equal(t, 1., tensor.Get(0, 0))
	require.Equal(t, 2., tensor.Get(0, 1))
	require.Equal(t, 3., tensor.Get(1, 0))
	require.Equal(t, 7., tensor.Get(2, 1))

	tensor, err := New([]int{2, 2, 2}, []float64{0, 1, 2, 3, 4, 5, 6, 7})
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

func TestIndex(t *testing.T) {
	t.Parallel()

	tensor, err := New([]int{2, 2}, []float64{0, 1, 2, 3})
	require.NoError(t, err)

	subTensor, err := tensor.Index(Index(0))
	require.NoError(t, err)
	require.Equal(t, []float64{0, 1}, subTensor.Items())

	subTensor, err = tensor.Index(Index(1))
	require.NoError(t, err)
	require.Equal(t, []float64{2, 3}, subTensor.Items())

	tensor, err = New([]int{2, 2, 2}, []float64{0, 1, 2, 3, 4, 5, 6, 7})
	require.NoError(t, err)

	subTensor, err = tensor.Index(Index(0))
	require.NoError(t, err)
	require.Equal(t, []float64{0, 1, 2, 3}, subTensor.Items())

	subTensor, err = tensor.Index(Index(1))
	require.NoError(t, err)
	require.Equal(t, []float64{4, 5, 6, 7}, subTensor.Items())

}
