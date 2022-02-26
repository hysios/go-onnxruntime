package utils

import (
	"log"
	"reflect"
	"unsafe"
)

func tensorSize(val interface{}, idx int) int {
	var v = reflect.ValueOf(val)
	v = reflect.Indirect(v)
	t := v.Type()
	for i := 0; t.Kind() == reflect.Array || t.Kind() == reflect.Slice; i++ {
		if i == idx {
			return v.Len()
		}

		t = t.Elem()
		if v.Len() == 0 {
			return 0
		} else {
			v = v.Index(0)
		}
	}

	return 0
}

func tensorShape(val interface{}) []int64 {
	var v = reflect.ValueOf(val)
	v = reflect.Indirect(v)
	var shape []int64
	for i := 0; v.Type().Kind() == reflect.Array || v.Type().Kind() == reflect.Slice; i++ {
		shape = append(shape, int64(v.Len()))
		v = v.Index(0)
	}
	return shape
}

func tensorFlat(val interface{}, ptr uintptr) []float32 {
	var (
		shapes = tensorShape(val)
		s      int
	)

	if len(shapes) > 0 {
		s = int(shapes[0])
	} else {
		return nil
	}

	v := reflect.ValueOf(val)
	if v.Kind() != reflect.Ptr {
		panic("must val ptr")
	}

	v = reflect.Indirect(v)

	for _, shap := range shapes[1:] {
		s *= int(shap)
	}

	head := (*reflect.SliceHeader)(unsafe.Pointer(v.UnsafeAddr()))
	log.Printf("head %v", head)
	var flatHead = &reflect.SliceHeader{
		Data: ptr,
		Len:  s,
		Cap:  s,
	}

	return *(*[]float32)(unsafe.Pointer(flatHead))
}

func tensorVal(val interface{}, index []int) interface{} {
	var v = reflect.ValueOf(val)
	v = reflect.Indirect(v)
	t := v.Type()
	for i := 0; t.Kind() == reflect.Array || t.Kind() == reflect.Slice; i++ {

		t = t.Elem()
		v = v.Index(index[i])
	}

	return v.Interface()
}

func tensorFlat2(val interface{}) []float32 {
	var (
		shapes = tensorShape(val)
		s      int
	)

	if len(shapes) > 0 {
		s = int(shapes[0])
	} else {
		return nil
	}

	for _, shap := range shapes[1:] {
		s *= int(shap)
	}

	var out = make([]float32, s)
	// [1,2,3]
	var index = make([]int, len(shapes))
	for i := 0; i < s; i++ {
		inc(index, shapes)

		out[dot(index)] = tensorVal(val, index).(float32)
	}
	return out
}

func reverse(shape []int64) []int64 {
	var re = make([]int64, len(shape))
	for i := 0; i < len(shape)/2; i++ {
		j := len(shape) - 1
		re[j], re[i] = shape[i], shape[j]
	}

	return re
}

func inc(indexs []int, shape []int64) {
	// shape =	reverse(shape)
	for i := len(shape) - 1; i >= 0; i-- {
		if indexs[i] < int(shape[i]) {
			indexs[i]++
		} else {
			continue
		}
	}
}

func dot(indexs []int) int {
	if len(indexs) == 0 {
		return 0
	}

	s := indexs[0]
	for _, m := range indexs[1:] {
		s *= m
	}
	return s
}

func CpuNMS(boxes [][]float32, scores []float32, threshold float32) []int {
	var (
		x1         = _select(boxes, 0)
		y1         = _select(boxes, 1)
		x2         = _select(boxes, 2)
		y2         = _select(boxes, 3)
		areas      = Area(x1, y1, x2, y2)
		order      = argsort(scores)
		num        = len(boxes)
		suppressed = make([]int, num)
		// keep       = make([][]float32, 0, num)
		keep = make([]int, 0, num)

		num_to_keep int
	)

	for _i := 0; _i < num; _i++ {
		i := order[_i]
		if suppressed[i] == 1 {
			continue
		}
		// keep = append(keep, boxes[i])
		keep = append(keep, i)
		// keep[num_to_keep] = i
		num_to_keep++
		var (
			ix1   = x1[i]
			iy1   = y1[i]
			ix2   = x2[i]
			iy2   = y2[i]
			iarea = areas[i]
		)

		for _j := _i + 1; _j < num; _j++ {
			j := order[_j]
			if suppressed[j] == 1 {
				continue
			}
			var (
				xx1   = max(ix1, x1[j])
				yy1   = max(iy1, y1[j])
				xx2   = min(ix2, x2[j])
				yy2   = min(iy2, y2[j])
				w     = max(0, xx2-xx1)
				h     = max(0, yy2-yy1)
				inter = w * h
				ovr   = inter / (iarea + areas[j] - inter)
			)

			if ovr > threshold {
				suppressed[j] = 1
			}
		}
	}

	return keep
}

func _select(boxes [][]float32,
	pos int) []float32 {
	var points = make([]float32, 0, len(boxes))

	for _, p := range boxes {
		points = append(points, p[pos])
	}

	return points
}
