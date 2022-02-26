package onnxruntime

import (
	"unsafe"

	"github.com/hysios/go-onnxruntime/internal/binding"
)

type Engine struct {
	cptr *binding.OrtApi
}

type RunOptions struct {
	cptr *binding.OrtRunOptions
}

func Open() (*Engine, error) {
	if baseApi == nil || core == nil {
		return nil, ErrNotInitialized
	}

	return &Engine{
		cptr: core.GetApi(baseApi),
	}, nil
}

func (api *Engine) CreateEnv(level OrtLoggingLevel, logid string) (*Env, error) {
	var (
		env    Env
		status = core.CreateEnv(api.cptr, binding.OrtLoggingLevel(level), binding.CString(logid), &env.cptr)
	)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &env, nil
}

func (api *Engine) CreateSessionOptions() (*SessionOptions, error) {
	var sessionOptions = SessionOptions{
		engine: api,
	}
	status := core.CreateSessionOptions(api.cptr, &sessionOptions.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &sessionOptions, nil
}

func (api *Engine) CreateSession(env *Env, modelPath string, optios *SessionOptions) (*Session, error) {
	var session = Session{
		engine: api,
	}

	status := core.CreateSession(api.cptr, env.cptr, modelPath, optios.cptr, &session.cptr)
	if status != nil {

		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}

	return &session, nil
}

func (api *Engine) CreateValue(val *Value, size int, typ ONNXType) (*Value, error) {
	var value Value
	status := core.CreateValue(api.cptr, &val.cptr, binding.Size(size), binding.ONNXType(typ), &value.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) CreateMemoryInfo(name string, typ OrtAllocatorType, id int, memType OrtMemType) (*MemoryInfo, error) {
	var value MemoryInfo
	status := core.CreateMemoryInfo(api.cptr, binding.CString(name), binding.OrtAllocatorType(typ), binding.Int32(id), binding.OrtMemType(memType), &value.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) CreateCpuMemoryInfo(typ OrtAllocatorType, memType OrtMemType) (*MemoryInfo, error) {
	var value MemoryInfo
	status := core.CreateCpuMemoryInfo(api.cptr, binding.OrtAllocatorType(typ), binding.OrtMemType(memType), &value.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) CreateAllocator(session *Session, memInfo *MemoryInfo) (*Allocator, error) {
	var allocator Allocator
	status := core.CreateAllocator(api.cptr, session.cptr, memInfo.cptr, &allocator.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &allocator, nil
}

func (api *Engine) CreateTensorAsOrtValue(allocator *Allocator, shape []int64, dataType ONNXTensorElementDataType) (*Value, error) {
	var value = Value{
		engine: api,
	}
	status := core.CreateTensorAsOrtValue(
		api.cptr,
		allocator.cptr,
		(*binding.Int64t)(unsafe.Pointer(&shape[0])),
		binding.Size(len(shape)),
		binding.ONNXTensorElementDataType(dataType),
		&value.cptr,
	)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) CreateTensorWithDataAsOrtValue(memInfo *MemoryInfo, p unsafe.Pointer, size int, shape []int64, dataType ONNXTensorElementDataType) (*Value, error) {
	var (
		value = Value{
			engine: api,
		}
	)

	status := core.CreateTensorWithDataAsOrtValue(
		api.cptr,
		memInfo.cptr,
		p,
		binding.Size(size),
		(*binding.Int64t)(unsafe.Pointer(&shape[0])),
		binding.Size(len(shape)),
		binding.ONNXTensorElementDataType(dataType),
		&value.cptr,
	)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) Allocator() (*Allocator, error) {
	var value Allocator

	status := core.GetAllocatorWithDefaultOptions(api.cptr, &value.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) CreateRunOptions() (*RunOptions, error) {
	var value RunOptions
	status := core.CreateRunOptions(api.cptr, &value.cptr)
	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	return &value, nil
}

func (api *Engine) Run(sess *Session, options *RunOptions, inputNames []string, inputs []*Value, outputNames []string) (outputs []*Value, err error) {
	var (
		_inputs_names  []*binding.Char
		_outputs_names []*binding.Char
		_inputs        []*binding.OrtValue
		_outputs       *binding.OrtValue
	)

	for _, name := range inputNames {
		_inputs_names = append(_inputs_names, binding.CString(name))
	}

	for _, name := range outputNames {
		_outputs_names = append(_outputs_names, binding.CString(name))
	}

	for _, input := range inputs {
		_inputs = append(_inputs, input.cptr)
	}

	// for _, output := range outputs {
	// 	_outputs = append(_outputs, output.cptr)
	// }

	status := core.Run(
		api.cptr,
		sess.cptr,
		options.cptr,
		(**binding.Char)(unsafe.Pointer(&_inputs_names[0])),
		&_inputs[0],
		(binding.Size)(len(_inputs_names)),
		(**binding.Char)(unsafe.Pointer(&_outputs_names[0])),
		(binding.Size)(len(_outputs_names)),
		&_outputs,
	)

	if status != nil {
		return nil, CodeErr(OrtErrorCode(core.GetStatus(api.cptr, status)))
	}
	p := (uintptr)(unsafe.Pointer(_outputs))

	for i := 0; i < len(outputNames); i++ {
		outputs = append(outputs, &Value{engine: api, cptr: (*binding.OrtValue)(unsafe.Pointer(p))})
		p += unsafe.Sizeof(_outputs)
	}

	return
}
