package onnxruntime

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

type Session struct {
	cptr *binding.OrtSession
}

type SessionOptions struct {
	cptr *binding.OrtSessionOptions
}

type TypeInfo struct {
	cptr *binding.OrtTypeInfo
}

type TensorTypeAndShapeInfo struct {
	cptr *binding.OrtTensorTypeAndShapeInfo
}

type SequenceTypeInfo struct {
	cptr *binding.OrtSequenceTypeInfo
}

type MapTypeInfo struct {
	cptr *binding.OrtMapTypeInfo
}

func (sess *Session) InputCount() int {
	var size binding.Size

	core.GetInputCount(DefaultEngine.cptr, sess.cptr, &size)
	return int(size)
}

func (sess *Session) GetInputName(index int) (string, error) {
	var name *binding.Char
	allocator, err := Alloc()
	if err != nil {
		return "", err
	}

	status := core.GetInputName(DefaultEngine.cptr, sess.cptr, binding.Size(index), allocator.cptr, &name)
	if status != nil {
		if status != nil {
			return "", CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
		}
	}

	return binding.GoString(name), nil
}

func (session *Session) InputNames() []string {
	var names []string
	for i := 0; i < session.InputCount(); i++ {
		name, err := session.GetInputName(i)
		if err != nil {
			continue
		}
		names = append(names, name)
	}
	return names
}

func (sess *Session) OutputCount() int {
	var size binding.Size
	core.GetOutputCount(DefaultEngine.cptr, sess.cptr, &size)
	return int(size)
}

func (sess *Session) GetOutputName(index int) (string, error) {
	var name *binding.Char
	allocator, err := DefaultEngine.Allocator()
	if err != nil {
		return "", err
	}

	status := core.GetOutputName(DefaultEngine.cptr, sess.cptr, binding.Size(index), allocator.cptr, &name)
	if status != nil {
		return "", CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return binding.GoString(name), nil
}

func (session *Session) OutputNames() []string {
	var names []string
	for i := 0; i < session.OutputCount(); i++ {
		name, err := session.GetOutputName(i)
		if err != nil {
			continue
		}
		names = append(names, name)
	}
	return names
}

func (sess *Session) GetInputType(index int) (*TypeInfo, error) {
	var typ = TypeInfo{}
	status := core.GetSessionInputType(DefaultEngine.cptr, sess.cptr, binding.Size(index), &typ.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return &typ, nil
}

func (sess *Session) GetOutputType(index int) (*TypeInfo, error) {
	var typ = TypeInfo{}
	status := core.GetSessionOutputType(DefaultEngine.cptr, sess.cptr, binding.Size(index), &typ.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}

	return &typ, nil
}

func (session *Session) InputsTypes() []*TypeInfo {
	var types []*TypeInfo
	for i := 0; i < session.InputCount(); i++ {
		typ, err := session.GetInputType(i)
		if err != nil {
			continue
		}
		types = append(types, typ)
	}
	return types
}

func (session *Session) OutputsTypes() []*TypeInfo {
	var types []*TypeInfo
	for i := 0; i < session.OutputCount(); i++ {
		typ, err := session.GetOutputType(i)
		if err != nil {
			continue
		}
		types = append(types, typ)
	}
	return types
}

func (session *Session) Release() {
	core.ReleaseSession(DefaultEngine.cptr, session.cptr)
}

func (sessOptions *SessionOptions) AppendExecutionProvider_CUDA(providerOptions *CUDAProviderOptions, device_id int) error {
	status, err := core.SessionOptionsAppendExecutionProvider_CUDA(sessOptions.cptr, providerOptions.cptr, device_id)
	if err != nil {
		return err
	}

	if status != nil {
		return CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))
	}
	return nil
}

func (sessOptions *SessionOptions) Release() {
	core.ReleaseSessionOptions(DefaultEngine.cptr, sessOptions.cptr)
}

func (ti *TypeInfo) OnnxType() ONNXType {
	var onnxType ONNXType
	core.GetOnnxTypeFromTypeInfo(DefaultEngine.cptr, ti.cptr, (*binding.ONNXType)(unsafe.Pointer(&onnxType)))
	return ONNXType(onnxType)
}

func (ti *TypeInfo) TensorShapeInfo() (*TensorTypeAndShapeInfo, error) {
	var tensor = TensorTypeAndShapeInfo{}

	status := core.CastTypeInfoToTensorInfo(DefaultEngine.cptr, ti.cptr, &tensor.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))

	}

	return &tensor, nil
}

// func (ti *TypeInfo) TensorSequenceInfo() (*SequenceTypeInfo, error) {
// 	var tensor = TensorTypeAndShapeInfo{}

// 	status := core.CastTypeInfoToTensorInfo(DefaultEngine.cptr, ti.cptr, &tensor.cptr)
// 	if status != nil {
// 		return nil, CodeErr(OrtErrorCode(core.GetStatus(DefaultEngine.cptr, status)))

// 	}

// 	return &tensor, nil
// }

func (ti *TypeInfo) String() string {
	var (
		tensorInfo string
	)
	tensor, err := ti.TensorShapeInfo()
	if err != nil {
		tensorInfo = fmt.Sprintf("[err: get tensor info %s]", err)
	} else {
		dims, err := tensor.Dimensions()
		if err != nil {
			tensorInfo = fmt.Sprintf("[err: get dimensions %s]", err)
		} else {
			tensorInfo = fmt.Sprintf("[")
			var dimss []string
			for _, d := range dims {
				dimss = append(dimss, strconv.Itoa(int(d)))
			}

			tensorInfo += strings.Join(dimss, "x") + "]"
		}
	}

	return fmt.Sprintf("TypeInfo[%p] OnnxType: %s Shape: %s ", ti, ti.OnnxType(), tensorInfo)
}
