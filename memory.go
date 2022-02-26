package onnxruntime

import "github.com/hysios/go-onnxruntime/internal/binding"

type Allocator struct {
	cptr *binding.OrtAllocator
}

type MemoryInfo struct {
	cptr *binding.OrtMemoryInfo
}
