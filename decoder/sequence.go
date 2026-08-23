package decoder

import (
	"github.com/cymertek/go-tokenizer"
)

// Sequence decodes tokens using multiple sub-decoders, one per sequence.
type Sequence struct {
	*DecoderBase

	decoders []tokenizer.Decoder
}

var _ tokenizer.Decoder = new(Sequence)

// NewSequence creates a new Sequence decoder with the given sub-decoders for each sequence.
func NewSequence(decoders []tokenizer.Decoder) *Sequence {
	base := new(DecoderBase)

	seq := &Sequence{
		DecoderBase: base,
		decoders:    decoders,
	}

	seq.Decoder = any(seq).(tokenizer.Decoder)

	return seq
}

// Decode implements `tokenizer.Decoder` interface.
func (d *Sequence) DecodeChain(tokens []string) []string {
	var input []string
	input = append(input, tokens...)
	for _, dec := range d.decoders {
		tmp := dec.DecodeChain(input)
		input = tmp
	}

	return input
}
