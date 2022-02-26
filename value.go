package onnxruntime

import (
	"reflect"
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

type Value struct {
	engine *Engine
	cptr   *binding.OrtValue
}

func (val *Value) Count() (int, error) {
	var size binding.Size
	status := core.GetValueCount(val.engine.cptr, val.cptr, &size)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(val.engine.cptr, status)))
	}

	return int(size), nil
}

func (val *Value) TensorData(size int) ([]float32, error) {
	var out *binding.Float
	status := core.GetTensorMutableData(val.engine.cptr, val.cptr, (*unsafe.Pointer)(unsafe.Pointer(&out)))
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(val.engine.cptr, status)))
	}

	var data []float32
	sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	sliceHeader.Cap = size
	sliceHeader.Len = size
	sliceHeader.Data = uintptr(unsafe.Pointer(out))

	output := append([]float32(nil), data...)
	return output, nil
}
