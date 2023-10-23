package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAdd(t *testing.T) {
	t.Parallel()

	t1, err := New(WithSize(3, 2, 3), WithValues(
		1, 2, 3,
		1, 2, 3,

		1, 2, 3,
		1, 2, 3,

		1, 2, 3,
		1, 2, 3,
	))
	require.NoError(t, err)

	t2, err := New(WithSize(3, 2, 3), WithValues(
		1, 2, 3,
		1, 2, 3,

		5, 2, 3,
		5, 2, 3,

		1, 2, 3,
		1, 2, 3,
	))
	require.NoError(t, err)

	expectedRes, err := New(WithSize(3, 2, 3), WithValues(
		2, 4, 6,
		2, 4, 6,

		6, 4, 6,
		6, 4, 6,

		2, 4, 6,
		2, 4, 6,
	))
	require.NoError(t, err)

	res, err := Add(t1, t2)
	require.NoError(t, err)
	require.Equal(t, expectedRes.Items(), res.Items())
}

func TestAddWithDifferentDimensions(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		t1            *MaybeTensor
		t2            *MaybeTensor
		expectedItems []float64
	}{
		{
			Wrap(New(WithSize(2), WithValues(2, 3))),
			Wrap(New(WithSize(3, 2), WithValues(2, 2, 3, 3, 6, 6))),
			[]float64{4, 5, 5, 6, 8, 9},
		},
		{
			Wrap(New(WithValues(0, 0, 0, 0))),
			Wrap(New(WithValues(1))),
			[]float64{1, 1, 1, 1},
		},
	}

	for _, tc := range testCases {
		res, err := tc.t1.Do(Add, tc.t2).Value()
		require.NoError(t, err)
		require.Equal(t, tc.expectedItems, res.Items())
	}
}

func TestBroadcasting(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		a                *Tensor
		b                *Tensor
		aExpectedStrides []int
		bExpectedStrides []int
		aExpectedSizes   []int
		bExpectedSizes   []int
	}{
		{
			a: &Tensor{
				sizes:   []int{5, 1, 5},
				strides: []int{5, 5, 1},
			},
			b: &Tensor{
				sizes:   []int{1, 5, 1},
				strides: []int{5, 1, 1},
			},
			aExpectedStrides: []int{5, 0, 1},
			aExpectedSizes:   []int{5, 5, 5},
			bExpectedStrides: []int{0, 1, 0},
			bExpectedSizes:   []int{5, 5, 5},
		},
		{
			a: &Tensor{
				sizes:   []int{5},
				strides: []int{1},
			},
			b: &Tensor{
				sizes:   []int{3, 4, 5},
				strides: []int{20, 5, 1},
			},
			aExpectedStrides: []int{0, 0, 1},
			aExpectedSizes:   []int{3, 4, 5},
			bExpectedStrides: []int{20, 5, 1},
			bExpectedSizes:   []int{3, 4, 5},
		},
	}

	for _, tc := range testCases {
		resA, resB, err := Broadcast(tc.a, tc.b)
		require.NoError(t, err)
		require.Equal(t, tc.aExpectedSizes, resA.sizes)
		require.Equal(t, tc.aExpectedStrides, resA.strides)
		require.Equal(t, tc.bExpectedSizes, resB.sizes)
		require.Equal(t, tc.bExpectedStrides, resB.strides)
	}
}
