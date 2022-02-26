package onnxruntime

import "testing"

func TestMain(m *testing.M) {
	Init()

	m.Run()
}

func TestVersion(t *testing.T) {

	t.Logf("version: %s", Version())
}
