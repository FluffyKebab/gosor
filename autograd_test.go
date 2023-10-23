package gosor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAutoGradAdd(t *testing.T) {
	t.Parallel()

	a := Wrap(New(WithValues(2), WithRecordGradients()))
	b := Wrap(New(WithValues(3), WithRecordGradients()))
	c := Wrap(New(WithValues(9), WithRecordGradients()))

	res, err := a.Do(Add, b).Do(Add, c).Value()
	require.NoError(t, err)
	require.Equal(t, 14., res.Item())

	err = res.Backward(nil)
	require.NoError(t, err)

	require.Equal(t, []float64{1}, res.gradient.Items())
	require.Equal(t, []float64{1}, c.MustValue().gradient.Items())
	require.Equal(t, []float64{1}, b.MustValue().gradient.Items())
	require.Equal(t, []float64{1}, a.MustValue().gradient.Items())
}

func TestAutoGradSub(t *testing.T) {
	t.Parallel()

	a := Wrap(New(WithValues(4), WithRecordGradients()))
	b := Wrap(New(WithValues(2), WithRecordGradients()))

	res, err := a.Do(Sub, b).Value()
	require.NoError(t, err)

	err = res.Backward(nil)
	require.NoError(t, err)

	require.Equal(t, []float64{2}, res.Items())
	require.Equal(t, []float64{1}, a.MustValue().gradient.Items())
	require.Equal(t, []float64{-1}, b.MustValue().gradient.Items())
}

func TestAutoGradPow(t *testing.T) {
	t.Parallel()

	a := Wrap(New(WithValues(4), WithRecordGradients()))
	b := Wrap(New(WithValues(2), WithRecordGradients()))

	res, err := a.Do(Pow, b).Value()
	require.NoError(t, err)
	require.Equal(t, []float64{16}, res.Items())

	err = res.Backward(nil)
	require.NoError(t, err)
	require.Equal(t, []float64{8}, a.MustValue().gradient.Items())
}

func TestGradSum(t *testing.T) {
	t.Parallel()

	a := Wrap(New(WithValues(1, 1, 1, 1), WithRecordGradients()))
	b, err := a.DoT(Sum).Value()
	require.NoError(t, err)
	require.Equal(t, []float64{4}, b.Items())

	fmt.Println(b.gradFunc())

	err = b.Backward(nil)
	require.NoError(t, err)
	fmt.Println("ja: ", a.MustValue().gradient)
	require.Equal(t, []float64{1, 1, 1, 1}, a.MustValue().gradient.Items())
}
