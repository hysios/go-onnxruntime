package onnxruntime

import (
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

func (tensor *TensorTypeAndShapeInfo) ElementType() (ONNXTensorElementDataType, error) {
	var elementDataType binding.ONNXTensorElementDataType
	status := core.GetTensorElementType(DefaultEngine.cptr, tensor.cptr, &elementDataType)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))

	}

	return ONNXTensorElementDataType(elementDataType), nil
}

func (tensor *TensorTypeAndShapeInfo) ElementCount() (int, error) {
	var size binding.Size
	status := core.GetTensorElementCount(DefaultEngine.cptr, tensor.cptr, &size)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return int(size), nil
}

func (tensor *TensorTypeAndShapeInfo) DimensionsCount() (int, error) {
	var size binding.Size
	status := core.GetTensorDimensionsCount(DefaultEngine.cptr, tensor.cptr, &size)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
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

	status := core.GetTensorDimensions(DefaultEngine.cptr, tensor.cptr, (*binding.Int64t)(unsafe.Pointer(&values[0])), binding.Size(count))
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return values, nil
}

func (tensor *TensorTypeAndShapeInfo) DataFloat32s() ([]float32, error) {
	count, err := tensor.DimensionsCount()
	if err != nil {
		return nil, err
	}

	var (
		raw unsafe.Pointer
	)

	status := core.GetTensorMutableData(DefaultEngine.cptr, (*binding.OrtValue)(tensor.cptr), &raw)

	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return unsafe.Slice((*float32)(unsafe.Pointer(raw)), count), nil
}
