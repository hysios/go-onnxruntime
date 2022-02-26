package binding

import "testing"

func TestT(t *testing.T) {
	api := CC.GetApiBase()
	t.Logf("onnxruntime version: %s", CC.GetVersionString(api))
}
