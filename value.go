package onnxruntime

import (
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

type Value struct {
	cptr *binding.OrtValue
}

func (val *Value) Count() (int, error) {
	var size binding.Size
	status := core.GetValueCount(DefaultEngine.cptr, val.cptr, &size)
	if status != nil {
		return 0, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return int(size), nil
}

func (val *Value) TensorData(size int) ([]float32, error) {
	var out *binding.Float
	status := core.GetTensorMutableData(DefaultEngine.cptr, val.cptr, (*unsafe.Pointer)(unsafe.Pointer(&out)))
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	// var data []float32
	// sliceHeader := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	// sliceHeader.Cap = size
	// sliceHeader.Len = size
	// sliceHeader.Data = uintptr(unsafe.Pointer(out))

	// output := append([]float32(nil), data...)

	return unsafe.Slice((*float32)(unsafe.Pointer(out)), size), nil
	// return output, nil
}

func (val *Value) GetTensorTypeAndShape() (*TensorTypeAndShapeInfo, error) {
	var out *binding.OrtTensorTypeAndShapeInfo

	status := core.GetTensorTypeAndShape(DefaultEngine.cptr, val.cptr, &out)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return &TensorTypeAndShapeInfo{cptr: out}, nil
}

func (val *Value) Release() {
	core.ReleaseValue(DefaultEngine.cptr, val.cptr)
}
