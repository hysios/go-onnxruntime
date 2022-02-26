package onnxruntime

import (
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

type Session struct {
	engine *Engine
	cptr   *binding.OrtSession
}

type SessionOptions struct {
	engine *Engine
	cptr   *binding.OrtSessionOptions
}

type TypeInfo struct {
	engine *Engine
	cptr   *binding.OrtTypeInfo
}

type TensorTypeAndShapeInfo struct {
	engine *Engine
	cptr   *binding.OrtTensorTypeAndShapeInfo
}

func (sess *Session) InputCount() int {

	var size binding.Size
	core.GetInputCount(sess.engine.cptr, sess.cptr, &size)
	return int(size)
}

func (sess *Session) GetInputName(index int) (string, error) {
	var name *binding.Char
	allocator, err := sess.engine.Allocator()
	if err != nil {
		return "", err
	}

	status := core.GetInputName(sess.engine.cptr, sess.cptr, binding.Size(index), allocator.cptr, &name)
	if status != nil {
		if status != nil {
			return "", CodeErr(OrtErrorCode(core.GetStatus(sess.engine.cptr, status)))
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
	core.GetOutputCount(sess.engine.cptr, sess.cptr, &size)
	return int(size)
}

func (sess *Session) GetOutputName(index int) (string, error) {
	var name *binding.Char
	allocator, err := sess.engine.Allocator()
	if err != nil {
		return "", err
	}

	status := core.GetOutputName(sess.engine.cptr, sess.cptr, binding.Size(index), allocator.cptr, &name)
	if status != nil {
		return "", CodeErr(OrtErrorCode(core.GetStatus(sess.engine.cptr, status)))
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
	var typ = TypeInfo{engine: sess.engine}
	status := core.GetSessionInputType(sess.engine.cptr, sess.cptr, binding.Size(index), &typ.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(sess.engine.cptr, status)))
	}

	return &typ, nil
}

func (sess *Session) GetOutputType(index int) (*TypeInfo, error) {
	var typ = TypeInfo{engine: sess.engine}
	status := core.GetSessionOutputType(sess.engine.cptr, sess.cptr, binding.Size(index), &typ.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(sess.engine.cptr, status)))
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

func (sessOptions *SessionOptions) AppendExecutionProvider_CUDA(providerOptions *CUDAProviderOptions, device_id int) error {
	status, err := core.SessionOptionsAppendExecutionProvider_CUDA(sessOptions.cptr, providerOptions.cptr, device_id)
	if err != nil {
		return err
	}

	if status != nil {
		return CodeErr(OrtErrorCode(core.GetStatus(sessOptions.engine.cptr, status)))
	}
	return nil
}

func (ti *TypeInfo) OnnxType() ONNXType {
	var onnxType ONNXType
	core.GetOnnxTypeFromTypeInfo(ti.engine.cptr, ti.cptr, (*binding.ONNXType)(unsafe.Pointer(&onnxType)))
	return ONNXType(onnxType)
}

func (ti *TypeInfo) TensorInfo() (*TensorTypeAndShapeInfo, error) {
	var tensor = TensorTypeAndShapeInfo{
		engine: ti.engine,
	}

	status := core.CastTypeInfoToTensorInfo(ti.engine.cptr, ti.cptr, &tensor.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(ti.engine.cptr, status)))

	}

	return &tensor, nil
}
