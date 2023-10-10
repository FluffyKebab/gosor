package gosor

import "errors"

var (
	ErrInvalidTensor    = errors.New("invalid tensor")
	ErrIndexOutOfBounds = errors.New("index out of bounds")
)
