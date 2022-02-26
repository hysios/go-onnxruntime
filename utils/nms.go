package utils

import (
	"log"
	"sort"
)

type PointInRectangle int

const (
	XMIN PointInRectangle = iota
	YMIN
	XMAX
	YMAX
	SCORE
)

func NMS(boxes [][]float32, threshold float32) [][]float32 {
	if len(boxes) == 0 {
		return [][]float32{}
	}

	var (
		x1    = GetPointFromRect(boxes, XMIN)
		y1    = GetPointFromRect(boxes, YMIN)
		x2    = GetPointFromRect(boxes, XMAX)
		y2    = GetPointFromRect(boxes, YMAX)
		score = GetPointFromRect(boxes, SCORE)
	)

	var (
		area = Area(x1, y1, x2, y2)
		idxs = argsort(score)
		pick []int
	)
	log.Printf("idxs %v", idxs)
	log.Printf("area %v", area)
	for len(idxs) > 0 {
		var (
			last = len(idxs) - 1
			i    = idxs[last]
		)
		pick = append(pick, i)

		var (
			idxsWoLast = RemoveLast(idxs)
			xx1        = Maximum(x1[i], CopyByIndexes(x1, idxsWoLast))
			yy1        = Maximum(y1[i], CopyByIndexes(y1, idxsWoLast))
			xx2        = Minimum(x2[i], CopyByIndexes(x2, idxsWoLast))
			yy2        = Minimum(y2[i], CopyByIndexes(y2, idxsWoLast))

			w = Maximum(0, Subtract(xx2, xx1))
			h = Maximum(0, Subtract(yy2, yy1))
		)

		// if len(w) == 0 || len(h) == 0 {
		// 	idxs = RemoveByIndexes(idxs, idxsWoLast)
		// 	continue
		// }

		log.Printf("w %v x h %v", w, h)
		log.Printf("sub w %v", Subtract(xx2, xx1))

		var (
			// overlap = Divide(Multiply(w, h), Subtract(Plus(area, CopyByIndexes(area, idxsWoLast)), Multiply(w, h)))
			overlap    = Divide(Multiply(w, h), CopyByIndexes(area, idxsWoLast))
			deleteIdxs = WhereLarger(overlap, threshold)
		)
		log.Printf("overlap %v", overlap)
		deleteIdxs = append(deleteIdxs, last)
		log.Printf("deleteIdxs %v", deleteIdxs)
		idxs = RemoveByIndexes(idxs, deleteIdxs)
	}

	return FilterVector(boxes, pick)
}

func GetPointFromRect(boxes [][]float32,
	pos PointInRectangle) []float32 {
	var points = make([]float32, 0, len(boxes))

	for _, p := range boxes {
		points = append(points, p[int(pos)])
	}

	return points
}

type argIntSort struct {
	values  []float32
	indexes []int
}

// Len is the number of elements in the collection.
func (args argIntSort) Len() int {
	return len(args.values)
}

func (args argIntSort) Less(i int, j int) bool {
	a, b := args.indexes[i], args.indexes[j]

	return args.values[a] > args.values[b]
}

// Swap swaps the elements with indexes i and j.
func (args argIntSort) Swap(i int, j int) {
	args.indexes[i], args.indexes[j] = args.indexes[j], args.indexes[i]
}

func argsort(arr []float32) []int {
	// initialize original index locations
	var idx = make([]int, len(arr))
	for i := range idx {
		idx[i] = i
	}
	s := argIntSort{arr, idx}
	sort.Sort(s)
	return idx
}

func Area(x1, y1, x2, y2 []float32) []float32 {
	var area = make([]float32, len(x1))
	if len(x1) != len(x2) || len(x1) != len(y1) || len(x1) != len(y2) {
		panic("mismatch slice length")
	}

	for i := 0; i < len(x1); i++ {
		area[i] = (x2[i] - x1[i]) * (y2[i] - y1[i])
	}

	return area
}

func Maximum(num float32,
	arr []float32) []float32 {
	var (
		maxVec = make([]float32, len(arr))
	)

	copy(maxVec, arr)

	for i := 0; i < len(maxVec); i++ {
		if arr[i] < num {
			maxVec[i] = num
		}
	}
	return maxVec
}

func Minimum(num float32,
	arr []float32) []float32 {
	var (
		maxVec = make([]float32, len(arr))
	)
	copy(maxVec, arr)
	for i := 0; i < len(maxVec); i++ {
		if arr[i] > num {
			maxVec[i] = num
		}
	}
	return maxVec
}

func CopyByIndexes(arr []float32,
	idxs []int) []float32 {
	var result = make([]float32, 0, len(idxs))

	for _, idx := range idxs {
		result = append(result, arr[idx])
	}

	return result
}

func RemoveLast(arr []int) []int {
	return arr[:len(arr)-1]
}

func Plus(a, b []float32) []float32 {
	var result = make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		if i < len(b)-1 {
			result[i] = a[i] + b[i]
		}
	}
	return result
}

func Subtract(a, b []float32) []float32 {
	var result = make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		if i < len(b)-1 {
			result[i] = a[i] - b[i] + 1
		}
	}
	return result
}

func Multiply(a, b []float32) []float32 {
	var result = make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] * b[i]
	}
	return result
}

func Divide(a, b []float32) []float32 {
	var result = make([]float32, len(a))
	for i := 0; i < len(a); i++ {
		result[i] = a[i] / b[i]
	}
	return result
}

func WhereLarger(a []float32, threshold float32) []int {
	var result = make([]int, 0, len(a))
	for i := 0; i < len(a); i++ {
		if a[i] > threshold {
			result = append(result, i)
		}
	}
	return result
}

func remove(slice []int, s int) []int {
	return append(slice[:s], slice[s+1:]...)
}

func RemoveByIndexes(a []int,
	idxs []int) []int {

	var (
		result = make([]int, len(a))
		offset int
	)
	copy(result, a)
	for _, idx := range idxs {
		copy(result[:idx+offset], result[idx+1+offset:])
		result = result[:len(result)-1]
		offset--
	}

	return result
}

func FilterVector(arr [][]float32, idxs []int) [][]float32 {
	var result = make([][]float32, 0, len(idxs))
	for _, idx := range idxs {
		result = append(result, arr[idx])
	}

	return result
}
