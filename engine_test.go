package onnxruntime

import (
	"errors"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"testing"
	"unsafe"

	"github.com/nfnt/resize"
	"github.com/tj/assert"
)

var logLevel = ORT_LOGGING_LEVEL_ERROR

func TestAPI_CreateEnv(t *testing.T) {
	api, err := Open()
	assert.NoError(t, err)

	env, err := api.CreateEnv(logLevel, "")
	assert.NoError(t, err)
	assert.NotNil(t, env)
	t.Logf("env %#v", env)
}

func TestAPI_CreateSession(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)

	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)
	t.Logf("env %#v", env)
	t.Logf("session %#v", session)
}

func TestSession_InputNames(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	t.Logf("inputs %v", session.InputNames())
}

func TestSession_OutputNames(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	t.Logf("outputs %v", session.OutputNames())
}

func TestSession_OutputTypes(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	t.Logf("outputs types % #v", session.OutputsTypes())
}

func TestSession_InputsTypes(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	t.Logf("inputs types % #v", session.InputsTypes())
}

func TestSession_InputsTypesOnnxType(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	for _, inputType := range session.InputsTypes() {
		t.Logf("inputType onnxType %v", inputType.OnnxType())
	}
}

func TestSession_InputsTextRec(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/ocr_det.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	for _, inputType := range session.InputsTypes() {
		t.Logf("inputType onnxType %v", inputType.OnnxType())
	}
}

func TestSession_InputsTypesTensorInfo(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	for i, inputType := range session.InputsTypes() {
		t.Logf("type %d ------------", i)
		tensor, err := inputType.TensorShapeInfo()
		assert.NoError(t, err)
		tensorDataType, err := tensor.ElementType()
		assert.NoError(t, err)
		t.Logf("tensor data type %v", tensorDataType)
		dimCount, err := tensor.DimensionsCount()
		assert.NoError(t, err)
		t.Logf("tensor dimensions count %v", dimCount)

		dims, err := tensor.Dimensions()
		assert.NoError(t, err)
		t.Logf("tensor dimensions %#v", dims)
	}
}

func TestSession_OutputsTypesTensorInfo(t *testing.T) {
	var (
		api, _     = Open()
		env, _     = api.CreateEnv(logLevel, "")
		options, _ = api.CreateSessionOptions()
	)
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	for _, inputType := range session.OutputsTypes() {
		tensor, err := inputType.TensorShapeInfo()
		assert.NoError(t, err)
		tensorDataType, err := tensor.ElementType()
		assert.NoError(t, err)
		t.Logf("tensor data type %v", tensorDataType)
		dimCount, err := tensor.DimensionsCount()
		assert.NoError(t, err)
		t.Logf("tensor dimensions count %v", dimCount)

		dims, err := tensor.Dimensions()
		assert.NoError(t, err)
		t.Logf("tensor dimensions %#v", dims)
	}
}

func TestSession_Run(t *testing.T) {
	var (
		api, _        = Open()
		env, _        = api.CreateEnv(logLevel, "")
		options, _    = api.CreateSessionOptions()
		memoryInfo, _ = api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeCPU)
		runOptions, _ = api.CreateRunOptions()
		//
	)
	// api.CreateCpuMemoryInfo()
	session, err := api.CreateSession(env, "./models/best.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)

	img, err := readImage("./images/demo.jpg", "jpg")
	assert.NoError(t, err)
	assert.NotNil(t, img)
	img = scale(img, 640, 640)
	var (
		p      = make([]float32, 3*640*640)
		result = make([][25200][15]float32, 1)
		mxp    = img.Bounds().Max
		c      int
	)
	for y := 0; y < mxp.Y; y++ {
		for x := 0; x < mxp.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			p[c] = float32(r)
			p[c+1] = float32(g)
			p[c+2] = float32(b)
			c += 3
		}
	}
	// img.At()

	input, err := api.CreateTensorWithDataAsOrtValue(memoryInfo, unsafe.Pointer(&p[0]), 3*640*640*4, []int64{1, 3, 640, 640}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	assert.NoError(t, err)
	assert.NotNil(t, input)
	t.Logf("input %v", input)
	// t.Log(input.Count())
	output, err := api.CreateTensorWithDataAsOrtValue(memoryInfo, unsafe.Pointer(&result[0]), 25200*15*4, []int64{1, 25200, 15}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	assert.NoError(t, err)
	assert.NotNil(t, output)
	t.Logf("rect %v", img.Bounds())
	t.Logf("output %v", output)
	t.Logf("options %v", runOptions)

	outputs, err := api.Run(session, runOptions, []string{"images"}, []*Value{input}, []string{"output"})
	assert.NoError(t, err)
	assert.NotNil(t, outputs)
	t.Log(output.Count())
	var indexes = make([]int, len(result[0]))
	// Argsort(result[0][:], indexes)
	t.Logf("indexes %v", indexes[:100])

	// output.
	t.Logf("result count %d", len(result[0]))
	t.Logf("result %v", result[0][:100])

}

func TestSession_RunTextRec(t *testing.T) {
	var (
		api, _        = Open()
		env, _        = api.CreateEnv(logLevel, "")
		options, _    = api.CreateSessionOptions()
		memoryInfo, _ = api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeCPU)
		runOptions, _ = api.CreateRunOptions()
		//
	)
	// api.CreateCpuMemoryInfo()
	session, err := api.CreateSession(env, "./models/ocr_det.onnx", options)
	assert.NoError(t, err)
	assert.NotNil(t, session)
	t.Logf("inputs names % #v", session.InputNames())
	t.Logf("outputs names % #v", session.OutputNames())

	for _, inputType := range session.OutputsTypes() {
		tensor, err := inputType.TensorShapeInfo()
		assert.NoError(t, err)
		tensorDataType, err := tensor.ElementType()
		assert.NoError(t, err)
		t.Logf("tensor data type %v", tensorDataType)
		dimCount, err := tensor.DimensionsCount()
		assert.NoError(t, err)
		t.Logf("tensor dimensions count %v", dimCount)

		dims, err := tensor.Dimensions()
		assert.NoError(t, err)
		t.Logf("tensor dimensions %#v", dims)
	}

	img, err := readImage("./images/车牌.png", "png")
	assert.NoError(t, err)
	assert.NotNil(t, img)
	img = scale(img, 320, 32)
	var (
		p      = make([]float32, 3*32*320)
		result = make([][25200][15]float32, 1)
		mxp    = img.Bounds().Max
		c      int
	)

	for y := 0; y < mxp.Y; y++ {
		for x := 0; x < mxp.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			p[c] = float32(r)
			p[c+1] = float32(g)
			p[c+2] = float32(b)
			c += 3
		}
	}
	// img.At()

	input, err := api.CreateTensorWithDataAsOrtValue(memoryInfo, unsafe.Pointer(&p[0]), 3*32*320*4, []int64{1, 3, 32, 320}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	assert.NoError(t, err)
	assert.NotNil(t, input)
	t.Logf("input %v", input)
	// t.Log(input.Count())
	output, err := api.CreateTensorWithDataAsOrtValue(memoryInfo, unsafe.Pointer(&result[0]), 25200*15*4, []int64{1, 25200, 15}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
	assert.NoError(t, err)
	assert.NotNil(t, output)
	t.Logf("rect %v", img.Bounds())
	t.Logf("output %v", output)
	t.Logf("options %v", runOptions)

	outputs, err := api.Run(session, runOptions, []string{"input"}, []*Value{input}, []string{"output"})
	assert.NoError(t, err)
	assert.NotNil(t, outputs)
	outType, err := session.GetOutputType(0)
	if err != nil {
		t.Fatalf("get output type error %s", err)
	}

	tensor, err := outType.TensorShapeInfo()
	if err != nil {
		t.Fatalf("tensor info error %s", err)
	}

	dims, err := tensor.Dimensions()
	if err != nil {
		t.Fatalf("get dimensions error %s", err)
	}
	t.Logf("%v", dims)
	results, err := outputs[0].TensorData(int(32 * 320))
	if err != nil {
		t.Fatalf("get tensor data err %s", err)
	}

	var (
		h = 32
		w = 320
	)

	for y := 0; y < h; y++ {
		t.Logf("%v", results[y*w:y*w+w])
	}

	t.Logf("results %d", len(results))
}

func readImage(filename string, typ string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	switch typ {
	case "jpg":
		return jpeg.Decode(f)
	case "png":
		return png.Decode(f)
	default:
		return nil, errors.New("non implement")
	}
}

func scale(src image.Image, width, height uint) image.Image {
	// and preserve aspect ratio

	return resize.Resize(width, height, src, resize.Lanczos3)
}
