package binding

import "C"

import (
	"syscall"
	"unicode/utf16"
)

func CString(s string) *Char {
	return (*Char)(C.CString(s))
}

func GoString(chars *Char) string {
	return C.GoString((*C.char)(chars))
}

func UTF16FromString(s string) ([]uint16, error) {
	for i := 0; i < len(s); i++ {
		if s[i] == 0 {
			return nil, syscall.EINVAL
		}
	}

	return utf16.Encode([]rune(s + "\x00")), nil
}

// UTF16ToString returns the UTF-8 encoding of the UTF-16 sequence s,
// with a terminating NUL removed.
func UTF16ToString(s []uint16) string {
	for i, v := range s {
		if v == 0 {
			s = s[0:i]
			break
		}
	}
	return string(utf16.Decode(s))
}

func StringToUTF16(s string) []uint16 {
	a, err := UTF16FromString(s)
	if err != nil {
		panic("syscall: string with NUL passed to StringToUTF16")
	}
	return a
}

func StringToUTF16Ptr(s string) *uint16 { return &StringToUTF16(s)[0] }
