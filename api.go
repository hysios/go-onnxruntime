package onnxruntime

import (
	"github.com/hysios/go-onnxruntime/internal/binding"
)

var (
	baseApi *binding.OrtApiBase
	_api    *API
	core    *binding.Core
)

type API struct {
	cptr *binding.OrtApi
}

func Init() {

	core = binding.CC
	baseApi = core.GetApiBase()
}

func Version() string {
	return core.GetVersionString(baseApi)
}
