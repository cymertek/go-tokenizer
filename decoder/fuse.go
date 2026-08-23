package decoder

import (
	"strings"

	"github.com/cymertek/go-tokenizer"
)

// Fuse constructs a Fuse decoder that joins all tokens into one big string.
type Fuse struct {
	*DecoderBase
}

// NewFuse creates a new Fuse decoder with default base decoder behavior.
func NewFuse() *Fuse {
	base := new(DecoderBase)

	d := &Fuse{base}

	d.Decoder = any(d).(tokenizer.Decoder)

	return d
}

func (f *Fuse) DecodeChain(tokens []string) []string {
	str := strings.Join(tokens, "")

	return []string{str}
}
