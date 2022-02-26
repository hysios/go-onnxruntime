package onnxruntime

import "github.com/hysios/go-onnxruntime/internal/binding"

type Env struct {
	cptr *binding.OrtEnv
}
