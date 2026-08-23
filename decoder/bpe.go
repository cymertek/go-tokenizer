package decoder

import (
	"strings"

	"github.com/cymertek/go-tokenizer"
)

// BpeDecoder decodes original BPE tokenization by joining tokens and replacing word-end suffixes with whitespace.
type BpeDecoder struct {
	*DecoderBase

	suffix string
}

// NewBpeDecoder creates a new BpeDecoder that joins tokens and replaces the given suffix with whitespace.
func NewBpeDecoder(suffix string) *BpeDecoder {
	base := new(DecoderBase)
	d := &BpeDecoder{
		DecoderBase: base,
		suffix:      suffix,
	}

	d.Decoder = interface{}(d).(tokenizer.Decoder)

	return d
}

// DefaultBpeDecoder create a new BpeDecoder with default suffix (`</w>`)
func DefaultBpeDecoder() *BpeDecoder {
	return &BpeDecoder{suffix: "</w>"}
}

/*
func (bd *BpeDecoder) Decode(tokens []string) string {
	output := strings.Join(tokens, "")
	output = strings.ReplaceAll(output, bd.suffix, " ")

	return output
}
*/

func (bd *BpeDecoder) DecodeChain(tokens []string) []string {
	var toks []string
	for i, token := range tokens {
		replacement := " "
		if i == len(tokens)-1 {
			replacement = ""
		}

		tok := strings.ReplaceAll(token, bd.suffix, replacement)
		toks = append(toks, tok)
	}

	return toks
}
