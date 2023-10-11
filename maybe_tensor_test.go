package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMaybe(t *testing.T) {
	t.Parallel()

	tensor := Wrap(New(WithSize(2, 2), WithValues(1, 2, 3, 4)))
	result, err := tensor.Index(Index(0)).
		Do(Add, tensor.Index(Index(1))).
		Do(Mul, Wrap(New(WithValues(2, 2)))).Value()
	require.NoError(t, err)
	require.Equal(t, []float64{8, 12}, result.Items())
}
