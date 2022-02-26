package onnxruntime

import "github.com/hysios/go-onnxruntime/internal/binding"

type CUDAProviderOptions struct {
	cptr *binding.OrtCUDAProviderOptions
}
