//go:build !windows
// +build !windows

package binding

// #cgo pkg-config: libonnxruntime
// #include "../include/onnxruntime_c_api.h"
import "C"

func init() {
	CC = &Core{
		GetApiBase: func() *OrtApiBase {
			return (*OrtApiBase)(C.OrtGetApiBase())
		},
	}
}
