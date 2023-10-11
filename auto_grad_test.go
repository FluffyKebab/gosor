package gosor

/* import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAutoGrad(t *testing.T) {
	t.Parallel()

	input, err := New([]int{3}, []float64{1, 2, 3})
	require.NoError(t, err)

	bias, err := New([]int{3}, []float64{1, 2, 3})
	require.NoError(t, err)
	bias.TrackGradients()

	desiredOutput, err := New([]int{3}, []float64{5, 0, 2})
	require.NoError(t, err)

	modelError := NewZeros(3)
	output := NewZeros(3)

	err = output.Add(input, bias)
	require.NoError(t, err)

	err = modelError.Min(desiredOutput, output)
	require.NoError(t, err)

	loss := Sum(modelError)

	loss
} */

/*
output, err := gosor.Dot(inputs, weights).Add(bias).Map(gosor.Relu).Unwrap()

*/
