package main

import (
	"bufio"
	"flag"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"github.com/hysios/go-onnxruntime"
	"github.com/hysios/go-onnxruntime/utils"
	"github.com/kr/pretty"
	"github.com/muesli/gamut"
	"github.com/nfnt/resize"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

var (
	input string
	label string
	model string
	nms   float64
)

func init() {
	flag.StringVar(&input, "input", "", "输入文件")
	flag.StringVar(&label, "label", "", "标签文件")
	flag.Float64Var(&nms, "nms", 0.45, "非极大值压制")

	flag.StringVar(&model, "model", "./models/yolov5s.onnx", "模型文件")
}

func main() {
	flag.Parse()
	onnxruntime.Init()
	if len(input) == 0 {
		flag.Usage()
		os.Exit(-1)
	}

	var (
		api, _        = onnxruntime.Open()
		env, _        = api.CreateEnv(onnxruntime.ORT_LOGGING_LEVEL_INFO, "")
		options, _    = api.CreateSessionOptions()
		memoryInfo, _ = api.CreateCpuMemoryInfo(onnxruntime.OrtDeviceAllocator, onnxruntime.OrtMemTypeCPU)
		runOptions, _ = api.CreateRunOptions()
		labels        []string
		//
	)
	// api.CreateCpuMemoryInfo()
	session, err := api.CreateSession(env, model, options)
	if err != nil {
		log.Fatalf("open model error %s", err)
	}

	img, err := readImage(input)
	if err != nil {
		log.Fatalf("read image error %s", err)
	}

	if len(label) == 0 {
		var (
			dir, filename = filepath.Split(model)
			ext           = filepath.Ext(filename)
			basename      = strings.TrimSuffix(filename, ext)
		)

		labels, err = readLabels(filepath.Join(dir, basename+".label.txt"))
		if err != nil {
			log.Fatalf("read labels %s error %s", filepath.Join(dir, basename+".label.txt"), err)
		}
	}

	colors, _ := gamut.Generate(len(labels), gamut.PastelGenerator{})
	for i, col := range colors {
		colors[i] = gamut.Lighter(col, 0.3)
	}

	img, _, _ = letterbox(img, Size{640, 640}, color.RGBA{114, 114, 114, 255}, false, 32)
	// img = scale(img, 640, 640)
	var (
		p   [3][640][640]float32
		mxp = img.Bounds().Max
		// c   int
	)

	for y := 0; y < mxp.Y; y++ {
		for x := 0; x < mxp.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// log.Printf("r %f, g %f,b %f", r, g, b)
			p[0][y][x] = float32(r>>8) / 255
			p[1][y][x] = float32(g>>8) / 255
			p[2][y][x] = float32(b>>8) / 255
		}
	}
	// log.Printf("input %v", p[200:400])
	// img.At()
	inputTensor, err := api.CreateTensorWithDataAsOrtValue(memoryInfo,
		unsafe.Pointer(&p[0]),
		3*640*640*4,
		[]int64{1, 3, 640, 640},
		onnxruntime.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
	)
	if err != nil {
		log.Fatalf("create tensor error %s", err)
	}
	log.Printf("input %v", inputTensor)

	outputs, err := api.Run(session, runOptions, []string{"images"}, []*onnxruntime.Value{inputTensor}, []string{"output"})
	if err != nil {
		log.Fatalf("run error %s", err)
	}
	log.Printf("outputs % #v", outputs)
	outType, err := session.GetOutputType(0)
	if err != nil {
		log.Fatalf("get output type error %s", err)
	}

	tensor, err := outType.TensorInfo()
	if err != nil {
		log.Fatalf("tensor info error %s", err)
	}

	dims, err := tensor.Dimensions()
	if err != nil {
		log.Fatalf("get dimensions error %s", err)
	}
	log.Printf("%v", dims)
	var (
		boxH    = dims[1]
		boxW    = dims[2]
		boxSize = boxH * boxW
	)
	results, err := outputs[0].TensorData(int(boxH * boxW))
	if err != nil {
		log.Fatalf("get tensor data err %s", err)
	}

	log.Printf("results %v", results[:1000])
	var (
		dir, filename = filepath.Split(input)
		ext           = filepath.Ext(filename)
		base          = filename[:len(filename)-len(ext)]
		outfile       = filepath.Join(dir, base+".out"+ext)
	)

	dst := image.NewNRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, image.Point{}, draw.Over)

	var (
		boxes  = make([][]float32, 0, 1000)
		boxes2 = make([][]float32, 0, 1000)
		scores []float32
		rects  [][]float32
		mx     = 4096
	)

	for i := 0; i < int(boxSize); i += int(boxW) {
		var result = make([]float32, boxW)
		copy(result[:], results[i:i+int(boxW)])
		if result[4] <= 0.25 {
			continue
		}
		cx, cy, w, h := result[0], result[1], result[2], result[3]
		x := cx - w/2
		y := cy - h/2
		c := getLabel(result[5:])
		result[0] = x + float32(c*mx)
		result[1] = y + float32(c*mx)
		result[2] = x + w + float32(c*mx)
		result[3] = y + h + float32(c*mx)
		boxes2 = append(boxes2, []float32{x, y, x + w, y + h, float32(c)})
		boxes = append(boxes, result)
		scores = append(scores, result[4])
	}

	log.Printf("boxes %d % #v", len(boxes2), pretty.Formatter(boxes2))
	if nms > 0 {
		index := utils.CpuNMS(boxes, scores, float32(nms))
		log.Printf("index %v", index)
		for _, idx := range index {
			rects = append(rects, boxes2[idx])
		}
		// rects = ort.NMS(boxes, float32(nms))
	} else {
		rects = boxes
	}
	classes := int(boxW) - 5
	log.Printf("classes %d", classes)
	for i, rect := range rects {
		x1, y1, x2, y2 := rect[0], rect[1], rect[2], rect[3]
		// x := cx - w/2
		// y := cy - h/2
		log.Printf("rect %v", image.Rect(int(x1), int(y1), int(x1), int(y2)))
		// drawRect(dst, image.Rect(int(x1), int(y1), int(x2), int(y2)))
		log.Printf("i %d c %d", i, i%(classes-1))
		drawClassBox(dst, image.Rect(int(x1), int(y1), int(x2), int(y2)), labels[int(rect[4])], colors[i%classes])
	}

	w, err := os.OpenFile(outfile, os.O_CREATE|os.O_RDWR|os.O_TRUNC, os.ModePerm)
	if err != nil {
		log.Fatalf("create output file '%s' error %s", outfile, err)
	}
	defer w.Close()
	if err := jpeg.Encode(w, dst, nil); err != nil {
		log.Fatalf("encode image to jpeg error %s", err)
	}

	// t.Log(output.Count())
	// var indexes = make([]int, len(result[0]))
	// SortArgs(result[0][:], indexes)
}

func readImage(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	img, _, err := image.Decode(f)
	return img, err
}

func scale(src image.Image, width, height uint) image.Image {
	// and preserve aspect ratio

	return resize.Resize(width, height, src, resize.Lanczos3)
}

// func DrawRectangle(dst draw.Image, r image.Rectangle) *image.NRGBA {
// 	// dst := image.NewNRGBA(imgsrc.Bounds())
// 	image.New
// 	draw.Draw(dst, r, imgsrc, image.ZP, draw.Over)
// 	draw.Draw(dst, r.Bounds(), r, image.ZP.Add(image.Pt(r.Min.X, r.Min.Y)), draw.Over)
// 	return dst
// }

func drawRect(dst draw.Image, r image.Rectangle) {
	blue := color.RGBA{0, 0, 255, 255}

	draw.Draw(dst, image.Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y+1), &image.Uniform{blue}, image.ZP, draw.Src)
	draw.Draw(dst, image.Rect(r.Max.X-1, r.Min.Y, r.Max.X, r.Max.Y), &image.Uniform{blue}, image.ZP, draw.Src)
	draw.Draw(dst, image.Rect(r.Min.X, r.Max.Y-1, r.Max.X, r.Max.Y), &image.Uniform{blue}, image.ZP, draw.Src)
	draw.Draw(dst, image.Rect(r.Min.X, r.Min.Y, r.Min.X+1, r.Max.Y), &image.Uniform{blue}, image.ZP, draw.Src)
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func drawClassBox(dst draw.Image, r image.Rectangle, text string, color color.Color) {
	border := 2
	drawLabel(dst, r.Min.X, max(0, r.Min.Y-5), color, text)
	draw.Draw(dst, image.Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y+border), &image.Uniform{color}, image.ZP, draw.Src)
	draw.Draw(dst, image.Rect(r.Max.X-border, r.Min.Y, r.Max.X, r.Max.Y), &image.Uniform{color}, image.ZP, draw.Src)
	draw.Draw(dst, image.Rect(r.Min.X, r.Max.Y-border, r.Max.X, r.Max.Y), &image.Uniform{color}, image.ZP, draw.Src)
	draw.Draw(dst, image.Rect(r.Min.X, r.Min.Y, r.Min.X+border, r.Max.Y), &image.Uniform{color}, image.ZP, draw.Src)
}

func drawLabel(img draw.Image, x, y int, col color.Color, label string) {
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}

func getLabelName(arr []float32, labels []string) string {
	log.Printf("len arr %d, len labels %d", len(arr), len(labels))
	if len(arr) != len(labels) {
		panic("missing length arguments")
	}

	var (
		max float32
		c   int
	)
	for i, v := range arr {
		if v > max {
			max = v
			c = i
		}
	}

	return labels[c]
}

func getLabel(arr []float32) int {
	var (
		max float32
		c   int
	)
	for i, v := range arr {
		if v > max {
			max = v
			c = i
		}
	}

	return c
}

type Size struct {
	Width  int
	Height int
}

func letterbox(img image.Image, newShape Size, color color.Color, auto bool, stride int) (image.Image, int, int) {
	var shape = Size{
		Width:  img.Bounds().Dx(),
		Height: img.Bounds().Dy(),
	}

	r := math.Min(float64(newShape.Width)/float64(shape.Height), float64(newShape.Height)/float64(shape.Width))
	new_unpad := []int{int(math.Round(float64(shape.Width) * r)), int(math.Round(float64(shape.Height) * r))}
	dw, dh := newShape.Width-new_unpad[0], newShape.Height-new_unpad[1]
	if auto {
		dw, dh = dw%stride, dh%stride
	}
	dw = dw / 2
	dh = dh / 2

	if shape.Width != new_unpad[0] || shape.Height != new_unpad[1] {
		img = scale(img, uint(new_unpad[0]), uint(new_unpad[1]))
	}

	top, _ := int(math.Round(float64(dh)-0.1)), int(math.Round(float64(dh)+0.1))
	left, _ := int(math.Round(float64(dw)-0.1)), int(math.Round(float64(dw)+0.1))
	rect := image.Rect(0, 0, newShape.Width, newShape.Height)
	bg := image.NewNRGBA(rect)
	draw.Draw(bg, rect, image.NewUniform(color), image.Point{}, draw.Src)
	draw.Draw(bg, image.Rect(0, 0, new_unpad[0], new_unpad[1]).Add(image.Pt(left, top)), img, image.Point{}, draw.Over)
	return bg, dw, dh
}

func readLabels(filename string) ([]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	defer f.Close()
	s := bufio.NewScanner(f)
	s.Split(bufio.ScanLines)
	var lines []string
	for s.Scan() {
		lines = append(lines, s.Text())
	}

	return lines, nil
}
