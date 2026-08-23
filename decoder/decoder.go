package decoder

import (
	"strings"

	"github.com/cymertek/go-tokenizer"
)

// DecoderBase provides default Decode and DecodeChain implementations that delegate to an embedded Decoder interface.
type DecoderBase struct {
	tokenizer.Decoder // Embed Decoder interface here so that a struct that embed `DecoderBase` can overwrite it method.
}

// NewDecoderBase creates a new decoder base with the given decoder implementation.
func (d *DecoderBase) Decode(tokens []string) string {
	return strings.Join(d.Decoder.DecodeChain(tokens), "")
}

// NOTE: this method here for validating only! It will be overloaded if a struct embed `DecoderBase` overwrites it.
func (d *DecoderBase) DecodeChain(tokens []string) []string {
	panic("NotImplementedError")
}
