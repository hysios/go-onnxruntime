package onnxruntime

type (
	OrtErrorCode              int
	OrtLoggingLevel           int
	ONNXType                  int
	OrtAllocatorType          int
	OrtMemType                int
	ONNXTensorElementDataType int
)

const (
	ORT_OK OrtErrorCode = iota
	ORT_FAIL
	ORT_INVALID_ARGUMENT
	ORT_NO_SUCHFILE
	ORT_NO_MODEL
	ORT_ENGINE_ERROR
	ORT_RUNTIME_EXCEPTION
	ORT_INVALID_PROTOBUF
	ORT_MODEL_LOADED
	ORT_NOT_IMPLEMENTED
	ORT_INVALID_GRAPH
	ORT_EP_FAIL
)

const (
	ORT_LOGGING_LEVEL_VERBOSE OrtLoggingLevel = iota
	ORT_LOGGING_LEVEL_INFO
	ORT_LOGGING_LEVEL_WARNING
	ORT_LOGGING_LEVEL_ERROR
	ORT_LOGGING_LEVEL_FATAL
)

const (
	ONNX_TYPE_UNKNOWN ONNXType = iota
	ONNX_TYPE_TENSOR
	ONNX_TYPE_SEQUENCE
	ONNX_TYPE_MAP
	ONNX_TYPE_OPAQUE
	ONNX_TYPE_SPARSETENSOR
)

const (
	OrtInvalidAllocator OrtAllocatorType = iota - 1
	OrtDeviceAllocator
	OrtArenaAllocator
)

const (
	OrtMemTypeCPUInput  OrtMemType = -2                  // Any CPU memory used by non-CPU execution provider
	OrtMemTypeCPUOutput OrtMemType = -1                  // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
	OrtMemTypeCPU       OrtMemType = OrtMemTypeCPUOutput // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
	OrtMemTypeDefault   OrtMemType = 0                   // the default allocator for execution provider
)

const (
	// Copied from TensorProto::DataType
	// Currently, Ort doesn't support complex64, complex128
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED ONNXTensorElementDataType = iota
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT                               // maps to c type float
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8                               // maps to c type uint8_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8                                // maps to c type int8_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16                              // maps to c type uint16_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16                               // maps to c type int16_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32                               // maps to c type int32_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64                               // maps to c type int64_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING                              // maps to c++ type std::string
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
	ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE     // maps to c type double
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32     // maps to c type uint32_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64     // maps to c type uint64_t
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64  // complex with float32 real and imaginary components
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 // complex with float64 real and imaginary components
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16   // Non-IEEE floating-point format based on IEEE754 single-precision
)
