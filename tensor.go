package onnxruntime

import (
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

func (tensor *TensorTypeAndShapeInfo) ElementType() (ONNXTensorElementDataType, error) {
	var elementDataType binding.ONNXTensorElementDataType
	status := core.GetTensorElementType(tensor.engine.cptr, tensor.cptr, &elementDataType)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(tensor.engine.cptr, status)))

	}

	return ONNXTensorElementDataType(elementDataType), nil
}

func (tensor *TensorTypeAndShapeInfo) ElementCount() (int, error) {
	var size binding.Size
	status := core.GetTensorElementCount(tensor.engine.cptr, tensor.cptr, &size)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(tensor.engine.cptr, status)))
	}

	return int(size), nil
}

func (tensor *TensorTypeAndShapeInfo) DimensionsCount() (int, error) {
	var size binding.Size
	status := core.GetTensorDimensionsCount(tensor.engine.cptr, tensor.cptr, &size)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(tensor.engine.cptr, status)))
	}

	return int(size), nil
}

func (tensor *TensorTypeAndShapeInfo) Dimensions() ([]int64, error) {
	count, err := tensor.DimensionsCount()
	if err != nil {
		return nil, err
	}

	var (
		values = make([]int64, count)
	)

	status := core.GetTensorDimensions(tensor.engine.cptr, tensor.cptr, (*binding.Int64t)(unsafe.Pointer(&values[0])), binding.Size(count))
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(tensor.engine.cptr, status)))
	}

	return values, nil
}
