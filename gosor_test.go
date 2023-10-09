package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTensorGettingIndex(t *testing.T) {
	t.Parallel()

	tensor := Zeros(3, 2)
	tensor.storage[0] = 1
	tensor.storage[1] = 2
	tensor.storage[2] = 3
	tensor.storage[5] = 7
	require.Equal(t, 1., tensor.Index(0, 0).Item())
	require.Equal(t, 2., tensor.Index(0, 1).Item())
	require.Equal(t, 3., tensor.Index(1, 0).Item())
	require.Equal(t, 7., tensor.Index(2, 1).Item())
}
