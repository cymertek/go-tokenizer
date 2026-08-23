package pretokenizer

import (
	"github.com/cymertek/go-tokenizer"
)

type Sequence struct {
	pretokenizers []tokenizer.PreTokenizer
}

var _ tokenizer.PreTokenizer = new(Sequence)

func NewSequence(pretokenizers []tokenizer.PreTokenizer) *Sequence {
	return &Sequence{pretokenizers}
}

// Implement tokenizer.PreTokenizer for Sequence

func (p *Sequence) PreTokenize(v *tokenizer.PreTokenizedString) (*tokenizer.PreTokenizedString, error) {
	out := v
	var err error
	for _, pretok := range p.pretokenizers {
		out, err = pretok.PreTokenize(out)
		if err != nil {
			return nil, err
		}
	}

	return out, nil
}
