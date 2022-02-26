package utils

func max_min(a, b float32) (float32, float32) {
	if a > b {
		return b, a
	} else {
		return a, b
	}
}

func max(a, b float32) float32 {
	if a > b {
		return a
	} else {
		return b
	}
}

func min(a, b float32) float32 {
	if a > b {
		return b
	} else {
		return a
	}
}
