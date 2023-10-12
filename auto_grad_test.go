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
