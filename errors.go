package gosor

import "errors"

var (
	ErrInvalidTensor         = errors.New("invalid tensor")
	ErrIndexOutOfBounds      = errors.New("index out of bounds")
	ErrUndefined             = errors.New("undefined")
	ErrFieldsMismatch        = errors.New("fields mismatch")
	ErrInvalidTensorCreation = errors.New("invalid tensor creation")
)
