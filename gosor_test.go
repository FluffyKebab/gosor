package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestItems(t *testing.T) {
	t.Parallel()

	var testCases = []struct {
		tensor        *Tensor
		expectedItems []float64
	}{
		{
			&Tensor{
				strides: []int{4, 1},
				sizes:   []int{3, 2},
				offset:  2,
				storage: []float64{
					0, 1, 2, 3,
					0, 1, 2, 3,
					0, 1, 2, 3,
				},
			},
			[]float64{2, 3, 2, 3, 2, 3},
		},
		{
			&Tensor{
				strides: []int{1},
				sizes:   []int{3},
				offset:  4,
				storage: []float64{
					0, 1, 2, 3,
					4, 5, 6, 7,
					8, 9, 10, 11,
				},
			},
			[]float64{4, 5, 6},
		},
		{
			&Tensor{
				strides: []int{4, 1},
				sizes:   []int{2, 2},
				offset:  1,
				storage: []float64{
					0, 1, 2, 3,
					4, 5, 6, 7,
					8, 9, 10, 11,
				},
			},
			[]float64{1, 2, 5, 6},
		},
	}

	for _, tc := range testCases {
		require.Equal(t, tc.expectedItems, tc.tensor.Items())
	}
}

func TestMap(t *testing.T) {
	t.Parallel()

	tensor := Wrap(New(WithRange(0, 10, 1)))
	res, err := tensor.Map(func(f float64) float64 {
		return f*2 + 2
	}).Value()
	require.NoError(t, err)
	require.Equal(t, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}, res.Items())
}

func TestSum(t *testing.T) {
	t.Parallel()

	var testCases = []struct {
		t             *MaybeTensor
		expectedItems []float64
	}{
		{
			t:             Wrap(New(WithRange(0, 10, 1), WithSize(2, 5))),
			expectedItems: []float64{5, 7, 9, 11, 13},
		},
	}

	for _, tc := range testCases {
		res, err := tc.t.DoT(Sum).Value()
		require.NoError(t, err)
		require.Equal(t, tc.expectedItems, res.Items())
	}
}
