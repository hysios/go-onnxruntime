package onnxruntime

import "errors"

var (
	ErrNotInitialized = errors.New("onnxruntime: not initialized")
)

func CodeErr(code OrtErrorCode) error {
	switch code {
	case ORT_FAIL:
		return errors.New("ORT_FAIL")
	case ORT_INVALID_ARGUMENT:
		return errors.New("ORT_INVALID_ARGUMENT")
	case ORT_NO_SUCHFILE:
		return errors.New("ORT_NO_SUCHFILE")
	case ORT_NO_MODEL:
		return errors.New("ORT_NO_MODEL")
	case ORT_ENGINE_ERROR:
		return errors.New("ORT_ENGINE_ERROR")
	case ORT_RUNTIME_EXCEPTION:
		return errors.New("ORT_RUNTIME_EXCEPTION")
	case ORT_INVALID_PROTOBUF:
		return errors.New("ORT_INVALID_PROTOBUF")
	case ORT_MODEL_LOADED:
		return errors.New("ORT_MODEL_LOADED")
	case ORT_NOT_IMPLEMENTED:
		return errors.New("ORT_NOT_IMPLEMENTED")
	case ORT_INVALID_GRAPH:
		return errors.New("ORT_INVALID_GRAPH")
	case ORT_EP_FAIL:
		return errors.New("ORT_EP_FAIL")
	default:
		return errors.New("unknown")
	}
}
