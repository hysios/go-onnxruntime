//go:build windows
// +build windows

package binding

// #undef _WIN32
// #include "../include/onnxruntime_c_api.h"
import "C"

import (
	"errors"
	"fmt"
	"log"
	"syscall"
	"unsafe"
)

var (
	onnxruntime                                           syscall.Handle
	getAppBaseProc                                        uintptr
	ortSessionOptionsAppendExecutionProvider_CPUProc      uintptr
	ortSessionOptionsAppendExecutionProvider_CUDAProc     uintptr
	ortSessionOptionsAppendExecutionProvider_TENSORRTProc uintptr
)

func init() {
	_onnxruntime, err := syscall.LoadLibrary("onnxruntime.dll")
	if err != nil {
		log.Fatal("load onnxruntime.dll failed:", err)
	}
	onnxruntime = _onnxruntime

	if getAppBaseProc, err = syscall.GetProcAddress(onnxruntime, "OrtGetApiBase"); err != nil {
		log.Fatal("get OrtGetApiBase failed:", err)
	}

	if ortSessionOptionsAppendExecutionProvider_CPUProc, err = syscall.GetProcAddress(onnxruntime, "OrtSessionOptionsAppendExecutionProvider_CPU"); err != nil {
		log.Printf("get OrtSessionOptionsAppendExecutionProvider_CPU failed: %s", err)
	}

	if ortSessionOptionsAppendExecutionProvider_CUDAProc, err = syscall.GetProcAddress(onnxruntime, "OrtSessionOptionsAppendExecutionProvider_CUDA"); err != nil {
		log.Printf("get OrtSessionOptionsAppendExecutionProvider_CUDA failed: %s", err)
	}

	if ortSessionOptionsAppendExecutionProvider_TENSORRTProc, err = syscall.GetProcAddress(onnxruntime, "OrtSessionOptionsAppendExecutionProvider_Tensorrt"); err != nil {
		log.Printf("get OrtSessionOptionsAppendExecutionProvider_Tensorrt failed: %s", err)
	}

	CC = &Core{
		GetApiBase: func() *OrtApiBase {
			r1, _, err := syscall.Syscall(getAppBaseProc, 0, 0, 0, 0)
			if err != 0 {
				log.Fatal("get OrtGetApiBase failed:", err)
			}
			return (*OrtApiBase)(unsafe.Pointer(r1))
		},
		SessionOptionsAppendExecutionProvider_CUDA: func(sessionOptions *OrtSessionOptions, provider *OrtCUDAProviderOptions, deviceID int) (*OrtStatus, error) {
			if ortSessionOptionsAppendExecutionProvider_CUDAProc == 0 {

				return nil, errors.New("Provider CUDA is not supported")

			}

			r1, _, err := syscall.Syscall(ortSessionOptionsAppendExecutionProvider_CUDAProc, 2, uintptr(unsafe.Pointer(sessionOptions)), uintptr(deviceID), 0)
			if err != 0 {
				return nil, fmt.Errorf("call ortSessionOptionsAppendExecutionProvider_CUDAProc failed: %s", err)
			}

			return (*OrtStatus)(unsafe.Pointer(r1)), nil
		},
	}
}
