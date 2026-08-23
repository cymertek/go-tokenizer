package models

// BType represents a GGUF value type (binary).
type BType uint32

const (
	BTypeUint8   BType = 0
	BTypeInt8    BType = 1
	BTypeUint16  BType = 2
	BTypeInt16   BType = 3
	BTypeUint32  BType = 4
	BTypeInt32   BType = 5
	BTypeFloat32 BType = 6
	BTypeBool    BType = 7
	BTypeString  BType = 8
	BTypeArray   BType = 9
	BTypeUint64  BType = 10
	BTypeInt64   BType = 11
	BTypeFloat64 BType = 12
)
