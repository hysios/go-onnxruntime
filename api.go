package onnxruntime

import (
	"github.com/hysios/go-onnxruntime/internal/binding"
)

var (
	baseApi       *binding.OrtApiBase
	_api          *API
	core          = binding.CC
	DefaultEngine = &Engine{
		cptr: core.GetApi(baseApi),
	}
)

type API struct {
	cptr *binding.OrtApi
}

func Init() (err error) {
	baseApi = core.GetApiBase()
	DefaultEngine, err = Open()
	return
}

func Version() string {
	return core.GetVersionString(baseApi)
}
