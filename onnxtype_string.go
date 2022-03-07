// Code generated by "stringer -type=ONNXType"; DO NOT EDIT.

package onnxruntime

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[ONNX_TYPE_UNKNOWN-0]
	_ = x[ONNX_TYPE_TENSOR-1]
	_ = x[ONNX_TYPE_SEQUENCE-2]
	_ = x[ONNX_TYPE_MAP-3]
	_ = x[ONNX_TYPE_OPAQUE-4]
	_ = x[ONNX_TYPE_SPARSETENSOR-5]
}

const _ONNXType_name = "ONNX_TYPE_UNKNOWNONNX_TYPE_TENSORONNX_TYPE_SEQUENCEONNX_TYPE_MAPONNX_TYPE_OPAQUEONNX_TYPE_SPARSETENSOR"

var _ONNXType_index = [...]uint8{0, 17, 33, 51, 64, 80, 102}

func (i ONNXType) String() string {
	if i < 0 || i >= ONNXType(len(_ONNXType_index)-1) {
		return "ONNXType(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _ONNXType_name[_ONNXType_index[i]:_ONNXType_index[i+1]]
}
