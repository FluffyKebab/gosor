package gosor

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAutoGradAdd(t *testing.T) {
	t.Parallel()

	a := Wrap(New(WithValues(2)))
	b := Wrap(New(WithValues(3)))
	c := Wrap(New(WithValues(9)))

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

	a := Wrap(New(WithValues(4)))
	b := Wrap(New(WithValues(2)))

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

	a := Wrap(New(WithValues(4)))
	b := Wrap(New(WithValues(2)))

	res, err := a.Do(Pow, b).Value()
	require.NoError(t, err)
	require.Equal(t, []float64{16}, res.Items())

	err = res.Backward(nil)
	require.NoError(t, err)
	require.Equal(t, []float64{8}, a.MustValue().gradient.Items())
}
