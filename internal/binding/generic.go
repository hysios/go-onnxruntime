package binding

// #undef _WIN32
// #include "binding.h"
import "C"

import (
	"runtime"
	"unsafe"
)

var CC = &Core{}

type (
	OrtApiBase                 C.OrtApiBase
	OrtApi                     C.OrtApi
	OrtStatus                  C.OrtStatus
	OrtErrorCode               C.OrtErrorCode
	OrtLoggingLevel            C.OrtLoggingLevel
	OrtEnv                     C.OrtEnv
	OrtSessionOptions          C.OrtSessionOptions
	OrtSession                 = C.OrtSession
	OrtValue                   C.OrtValue
	OrtTypeInfo                C.OrtTypeInfo
	ONNXType                   C.enum_ONNXType
	OrtMemType                 C.enum_OrtMemType
	OrtAllocatorType           C.OrtAllocatorType
	OrtMemoryInfo              C.OrtMemoryInfo
	OrtAllocator               C.OrtAllocator
	ONNXTensorElementDataType  C.ONNXTensorElementDataType
	OrtTensorTypeAndShapeInfo  C.OrtTensorTypeAndShapeInfo
	OrtSequenceTypeInfo        C.OrtSequenceTypeInfo
	OrtMapTypeInfo             C.OrtMapTypeInfo
	OrtRunOptions              C.OrtRunOptions
	OrtCUDAProviderOptions     C.OrtCUDAProviderOptions
	OrtROCMProviderOptions     C.OrtROCMProviderOptions
	OrtTensorRTProviderOptions C.OrtTensorRTProviderOptions
	OrtOpenVINOProviderOptions C.OrtOpenVINOProviderOptions
	Char                       = C.char
	Size                       = C.size_t
	Int32                      = C.int
	Int64t                     C.int64_t
	ULong                      C.ulong
	Float                      C.float
	Float64                    C.double
)

var (
	_api  *OrtApi
	_base *OrtApiBase
)

type Core struct {
	GetApiBase                                 func() *OrtApiBase
	SessionOptionsAppendExecutionProvider_CUDA func(os *OrtSessionOptions, provider *OrtCUDAProviderOptions, device_id int) (*OrtStatus, error)
}

func (core *Core) GetVersionString(api *OrtApiBase) string {
	return C.GoString(C.getVersionString((*C.OrtApiBase)(api)))
}

func (core *Core) GetApi(apiBase *OrtApiBase) *OrtApi {
	if apiBase == nil {
		apiBase = core.GetApiBase()
	}

	if _api == nil {
		_api = (*OrtApi)(C.getApi((*C.OrtApiBase)(apiBase)))
	}

	return _api
}

func (core *Core) CreateStatus(code OrtErrorCode, msg string) *OrtStatus {
	return (*OrtStatus)(C.createStatus((*C.OrtApi)(core.GetApi(nil)), C.OrtErrorCode(code), C.CString(msg)))
}

func (core *Core) GetStatus(api *OrtApi, os *OrtStatus) OrtErrorCode {
	return OrtErrorCode(C.getStatus((*C.OrtApi)(api), (*C.OrtStatus)(os)))
}

func (core *Core) CreateEnv(api *OrtApi, lvl OrtLoggingLevel, logid *Char, out **OrtEnv) *OrtStatus {
	var out1 *C.OrtEnv
	defer func() {
		*out = (*OrtEnv)(out1)
	}()
	return (*OrtStatus)(C.ortCreateEnv((*C.OrtApi)(api), C.OrtLoggingLevel(lvl), (*C.char)(logid), (**C.OrtEnv)(&out1)))
}

func (core *Core) CreateSessionOptions(api *OrtApi, out **OrtSessionOptions) *OrtStatus {
	var out1 *C.OrtSessionOptions
	defer func() {
		*out = (*OrtSessionOptions)(out1)
	}()
	return (*OrtStatus)(C.ortCreateSessionOptions((*C.OrtApi)(api), (**C.OrtSessionOptions)(&out1)))
}

func (core *Core) CreateSession(api *OrtApi, oe *OrtEnv, modelPath string, os *OrtSessionOptions, out **OrtSession) *OrtStatus {
	var out1 *C.OrtSession
	defer func() {
		*out = (*OrtSession)(out1)
	}()

	var cpath = (*C.char)(CString(modelPath))
	if runtime.GOOS == "windows" {
		cpath = (*C.char)(unsafe.Pointer(StringToUTF16Ptr(modelPath)))
	}

	return (*OrtStatus)(C.ortCreateSession((*C.OrtApi)(api), (*C.OrtEnv)(oe), cpath, (*C.OrtSessionOptions)(os), (**C.OrtSession)(&out1)))
}

func (core *Core) CreateSessionFromArray(api *OrtApi, oe *OrtEnv, model_data []byte, os *OrtSessionOptions, out **OrtSession) *OrtStatus {
	var out1 *C.OrtSession
	defer func() {
		*out = (*OrtSession)(out1)
	}()

	return (*OrtStatus)(C.ortCreateSessionFromArray(
		(*C.OrtApi)(api),
		(*C.OrtEnv)(oe),
		(unsafe.Pointer(&model_data[0])),
		(C.size_t)(len(model_data)),
		(*C.OrtSessionOptions)(os),
		(**C.OrtSession)(&out1)),
	)
}

func (core *Core) EnableProfiling(api *OrtApi, sess *OrtSessionOptions, prefix string) *OrtStatus {
	return (*OrtStatus)(C.ortEnableProfiling((*C.OrtApi)(api), (*C.OrtSessionOptions)(sess), C.CString(prefix)))
}

func (core *Core) DisableProfilng(api *OrtApi, sess *OrtSessionOptions) *OrtStatus {
	return (*OrtStatus)(C.ortDisableProfiling((*C.OrtApi)(api), (*C.OrtSessionOptions)(sess)))
}

func (core *Core) SessionEndProfiling(api *OrtApi, sess *OrtSession, allocator *OrtAllocator) (string, *OrtStatus) {
	var out *C.char

	status := (*OrtStatus)(C.ortSessionEndProfiling((*C.OrtApi)(api), (*C.OrtSession)(sess), (*C.OrtAllocator)(allocator), &out))
	return C.GoString(out), status
}

// func (core *Core) SessionOptionsAppendExecutionProvider_CUDA(api *OrtApi, os *OrtSessionOptions, provider *OrtCUDAProviderOptions, device_id int) *OrtStatus {
// 	// return (*OrtStatus)(C.ortSessionOptionsAppendExecutionProvider_CUDA((*C.OrtApi)(api), (*C.OrtSessionOptions)(os), (*C.OrtCUDAProviderOptions)(provider)))
// 	return (*OrtStatus)(C.ortSessionOptionsAppendExecutionProvider_CUDA((*C.OrtSessionOptions)(os), (*C.OrtCUDAProviderOptions)(provider), (*C.int)(device_id)))
// }

func (core *Core) CreateValue(oa *OrtApi, in **OrtValue, size Size, onnxType ONNXType, out **OrtValue) *OrtStatus {
	return (*OrtStatus)(C.ortCreateValue((*C.OrtApi)(oa), (**C.OrtValue)(unsafe.Pointer(in)), C.size_t(size), C.enum_ONNXType(onnxType), (**C.OrtValue)(unsafe.Pointer(out))))
}

func (core *Core) CreateMemoryInfo(oa *OrtApi, name *Char, allocType OrtAllocatorType, id Int32, memType OrtMemType, out **OrtMemoryInfo) *OrtStatus {
	return (*OrtStatus)(C.ortCreateMemoryInfo((*C.OrtApi)(oa), (*C.char)(name), C.OrtAllocatorType(allocType), C.int32_t(id), C.enum_OrtMemType(memType), (**C.OrtMemoryInfo)(unsafe.Pointer(out))))
}

func (core *Core) CreateCpuMemoryInfo(oa *OrtApi, allocType OrtAllocatorType, memType OrtMemType, out **OrtMemoryInfo) *OrtStatus {
	return (*OrtStatus)(C.ortCreateCpuMemoryInfo((*C.OrtApi)(oa), C.OrtAllocatorType(allocType), C.enum_OrtMemType(memType), (**C.OrtMemoryInfo)(unsafe.Pointer(out))))
}

func (core *Core) CreateAllocator(oa *OrtApi, sess *OrtSession, memInfo *OrtMemoryInfo, out **OrtAllocator) *OrtStatus {
	return (*OrtStatus)(C.ortCreateAllocator((*C.OrtApi)(oa), (*C.OrtSession)(sess), (*C.OrtMemoryInfo)(memInfo), (**C.OrtAllocator)(unsafe.Pointer(out))))
}

func (core *Core) CreateTensorAsOrtValue(oa *OrtApi, alloc *OrtAllocator, shape *Int64t, shapeLen Size, tensorTyp ONNXTensorElementDataType, out **OrtValue) *OrtStatus {
	return (*OrtStatus)(C.ortCreateTensorAsOrtValue((*C.OrtApi)(oa), (*C.OrtAllocator)(alloc), (*C.int64_t)(shape), C.size_t(shapeLen), C.ONNXTensorElementDataType(tensorTyp), (**C.OrtValue)(unsafe.Pointer(out))))
}

func (core *Core) CreateTensorWithDataAsOrtValue(oa *OrtApi, memInfo *OrtMemoryInfo, pData unsafe.Pointer, dataLen Size, shape *Int64t, shapeLen Size, tensorType ONNXTensorElementDataType, out **OrtValue) *OrtStatus {
	var out1 *C.OrtValue
	defer func() {
		*out = (*OrtValue)(out1)
	}()
	return (*OrtStatus)(C.ortCreateTensorWithDataAsOrtValue((*C.OrtApi)(oa), (*C.OrtMemoryInfo)(memInfo), pData, C.size_t(dataLen), (*C.int64_t)(shape), C.size_t(shapeLen), C.ONNXTensorElementDataType(tensorType), (**C.OrtValue)(&out1)))
}

func (core *Core) GetAllocatorWithDefaultOptions(api *OrtApi, out **OrtAllocator) *OrtStatus {
	return (*OrtStatus)(C.ortGetAllocatorWithDefaultOptions((*C.OrtApi)(api), (**C.OrtAllocator)(unsafe.Pointer(out))))
}

func (core *Core) CreateRunOptions(oa *OrtApi, out **OrtRunOptions) *OrtStatus {
	return (*OrtStatus)(C.ortCreateRunOptions((*C.OrtApi)(oa), (**C.OrtRunOptions)(unsafe.Pointer(out))))
}

func (core *Core) Run(oa *OrtApi, sess *OrtSession, options *OrtRunOptions, inputNames **Char, input **OrtValue, inputLen Size, outputNames **Char, outputLen Size, out **OrtValue) *OrtStatus {
	var out1 *C.OrtValue
	defer func() {
		*out = (*OrtValue)(out1)
	}()

	return (*OrtStatus)(C.ortRun(
		(*C.OrtApi)(oa),
		(*C.OrtSession)(sess),
		(*C.OrtRunOptions)(options),
		(**C.char)(unsafe.Pointer(inputNames)),
		(**C.OrtValue)(unsafe.Pointer(input)),
		C.size_t(inputLen),
		(**C.char)(unsafe.Pointer(outputNames)),
		C.size_t(outputLen),
		(**C.OrtValue)(&out1)))
}

func (core *Core) GetInputCount(api *OrtApi, session *OrtSession, out_count *Size) *OrtStatus {
	return (*OrtStatus)(C.ortSessionGetInputCount((*C.OrtApi)(api), (*C.OrtSession)(session), (*C.size_t)(unsafe.Pointer(out_count))))
}

func (core *Core) GetInputName(api *OrtApi, session *OrtSession, index Size, allocator *OrtAllocator, out **Char) *OrtStatus {
	return (*OrtStatus)(C.ortSessionGetInputName((*C.OrtApi)(api), (*C.OrtSession)(session), C.size_t(index), (*C.OrtAllocator)(allocator), (**C.char)(unsafe.Pointer(out))))
}

func (core *Core) GetOutputCount(api *OrtApi, session *OrtSession, out_count *Size) *OrtStatus {
	return (*OrtStatus)(C.ortSessionGetOutputCount((*C.OrtApi)(api), (*C.OrtSession)(session), (*C.size_t)(unsafe.Pointer(out_count))))
}

func (core *Core) GetOutputName(api *OrtApi, session *OrtSession, index Size, allocator *OrtAllocator, out **Char) *OrtStatus {
	return (*OrtStatus)(C.ortSessionGetOutputName((*C.OrtApi)(api), (*C.OrtSession)(session), C.size_t(index), (*C.OrtAllocator)(allocator), (**C.char)(unsafe.Pointer(out))))
}

func (core *Core) GetSessionInputType(api *OrtApi, session *OrtSession, index Size, out **OrtTypeInfo) *OrtStatus {
	var out1 *C.OrtTypeInfo

	defer func() {
		*out = (*OrtTypeInfo)(out1)
	}()

	return (*OrtStatus)(C.ortSessionGetInputTypeInfo((*C.OrtApi)(api), (*C.OrtSession)(session), C.size_t(index), (**C.OrtTypeInfo)(&out1)))
}

func (core *Core) GetSessionOutputType(api *OrtApi, session *OrtSession, index Size, out **OrtTypeInfo) *OrtStatus {
	var out1 *C.OrtTypeInfo
	defer func() {
		*out = (*OrtTypeInfo)(out1)
	}()
	return (*OrtStatus)(C.ortSessionGetOutputTypeInfo((*C.OrtApi)(api), (*C.OrtSession)(session), C.size_t(index), (**C.OrtTypeInfo)(&out1)))
}

func (core *Core) GetOnnxTypeFromTypeInfo(api *OrtApi, typeInfo *OrtTypeInfo, onnxType *ONNXType) *OrtStatus {
	return (*OrtStatus)(C.ortGetOnnxTypeFromTypeInfo((*C.OrtApi)(api), (*C.OrtTypeInfo)(typeInfo), (*C.enum_ONNXType)(unsafe.Pointer(onnxType))))
}

func (core *Core) GetOnnxTypeFromValue(*OrtApi, *OrtValue, *ONNXType) *OrtStatus {
	panic("not implemented")
}

func (core *Core) GetTensorElementType(api *OrtApi, tensor *OrtTensorTypeAndShapeInfo, out *ONNXTensorElementDataType) *OrtStatus {
	return (*OrtStatus)(C.ortGetTensorElementType((*C.OrtApi)(api), (*C.OrtTensorTypeAndShapeInfo)(tensor), (*uint32)(unsafe.Pointer(out))))
}

func (core *Core) GetTensorElementCount(api *OrtApi, tensor *OrtTensorTypeAndShapeInfo, out *Size) *OrtStatus {
	return (*OrtStatus)(C.ortGetTensorElementCount((*C.OrtApi)(api), (*C.OrtTensorTypeAndShapeInfo)(tensor), (*C.size_t)(unsafe.Pointer(out))))
}

func (core *Core) GetTensorDimensionsCount(api *OrtApi, tensor *OrtTensorTypeAndShapeInfo, out *Size) *OrtStatus {
	return (*OrtStatus)(C.ortGetTensorDimensionsCount((*C.OrtApi)(api), (*C.OrtTensorTypeAndShapeInfo)(tensor), (*C.size_t)(unsafe.Pointer(out))))
}

func (core *Core) GetTensorDimensions(api *OrtApi, in *OrtTensorTypeAndShapeInfo, dimValues *Int64t, out_len Size) *OrtStatus {
	return (*OrtStatus)(C.ortGetTensorDimensions((*C.OrtApi)(api), (*C.OrtTensorTypeAndShapeInfo)(in), (*C.int64_t)(unsafe.Pointer(dimValues)), (C.size_t)(out_len)))
}

func (core *Core) GetTensorData(*OrtApi, *OrtValue, Size, *ULong) *OrtStatus {
	panic("not implemented")
}

func (core *Core) GetTensorTypeAndShape(api *OrtApi, in *OrtValue, out **OrtTensorTypeAndShapeInfo) *OrtStatus {
	return (*OrtStatus)(C.ortGetTensorTypeAndShape((*C.OrtApi)(api), (*C.OrtValue)(in), (**C.OrtTensorTypeAndShapeInfo)(unsafe.Pointer(out))))
}

func (core *Core) CastTypeInfoToTensorInfo(api *OrtApi, typeInfo *OrtTypeInfo, out **OrtTensorTypeAndShapeInfo) *OrtStatus {
	var out1 *C.OrtTensorTypeAndShapeInfo
	defer func() {
		*out = (*OrtTensorTypeAndShapeInfo)(out1)
	}()
	return (*OrtStatus)(C.ortCastTypeInfoToTensorInfo((*C.OrtApi)(api), (*C.OrtTypeInfo)(typeInfo), (**C.OrtTensorTypeAndShapeInfo)(&out1)))
}

func (core *Core) CastTypeInfoToSequenceTypeInfo(api *OrtApi, typeInfo *OrtTypeInfo, out **OrtSequenceTypeInfo) *OrtStatus {
	var out1 *C.OrtSequenceTypeInfo
	defer func() {
		*out = (*OrtSequenceTypeInfo)(out1)
	}()

	return (*OrtStatus)(C.ortCastTypeInfoToSequenceTypeInfo((*C.OrtApi)(api), (*C.OrtTypeInfo)(typeInfo), (**C.OrtSequenceTypeInfo)(&out1)))
}

func (core *Core) CastTypeInfoToMapTypeInfo(api *OrtApi, typeInfo *OrtTypeInfo, out **OrtMapTypeInfo) *OrtStatus {
	var out1 *C.OrtMapTypeInfo
	defer func() {
		*out = (*OrtMapTypeInfo)(out1)
	}()

	return (*OrtStatus)(C.ortCastTypeInfoToMapTypeInfo((*C.OrtApi)(api), (*C.OrtTypeInfo)(typeInfo), (**C.OrtMapTypeInfo)(&out1)))
}

func (core *Core) GetTensorMutableData(api *OrtApi, value *OrtValue, out *unsafe.Pointer) *OrtStatus {
	return (*OrtStatus)(C.ortTensorMutableFloatData((*C.OrtApi)(api), (*C.OrtValue)(value), (*unsafe.Pointer)(out)))
}

func (core *Core) GetValueCount(api *OrtApi, value *OrtValue, out *Size) *OrtStatus {
	return (*OrtStatus)(C.ortGetValueCount((*C.OrtApi)(api), (*C.OrtValue)(value), (*C.size_t)(unsafe.Pointer(out))))
}

func (core *Core) ReleaseTensor(*OrtApi, *OrtValue) *OrtStatus { panic("not implemented") }
func (core *Core) ReleaseMemoryInfo(api *OrtApi, memInfo *OrtMemoryInfo) {
	C.ortReleaseMemoryInfo((*C.OrtApi)(api), (*C.OrtMemoryInfo)(memInfo))
}

func (core *Core) ReleaseSession(api *OrtApi, session *OrtSession) {
	C.ortReleaseSession((*C.OrtApi)(api), (*C.OrtSession)(session))
}

func (core *Core) ReleaseSessionOptions(api *OrtApi, sessionOptions *OrtSessionOptions) {
	C.ortReleaseSessionOptions((*C.OrtApi)(api), (*C.OrtSessionOptions)(sessionOptions))
}

func (core *Core) ReleaseEnv(api *OrtApi, env *OrtEnv) {
	C.ortReleaseEnv((*C.OrtApi)(api), (*C.OrtEnv)(env))
}

func (core *Core) ReleaseValue(api *OrtApi, value *OrtValue) {
	C.ortReleaseValue((*C.OrtApi)(api), (*C.OrtValue)(value))
}

func (core *Core) ReleaseRunOptions(api *OrtApi, options *OrtRunOptions) {
	C.ortReleaseRunOptions((*C.OrtApi)(api), (*C.OrtRunOptions)(options))
}
