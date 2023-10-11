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
