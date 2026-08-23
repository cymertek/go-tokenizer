package pretokenizer

import (
	// "fmt"
	// "unicode"

	"github.com/cymertek/go-tokenizer"
	"github.com/cymertek/go-tokenizer/normalizer"
)

func isBertPunc(x rune) (retVal bool) {
	// Use BERT's punctuation range table which matches HuggingFace tokenizer behavior.
	// This covers ASCII punctuation plus the BERT extension range defined in normalizer/bert.go.
	return normalizer.IsBertPunctuation(x)
}

type BertPreTokenizer struct{}

func NewBertPreTokenizer() *BertPreTokenizer {
	return &BertPreTokenizer{}
}

// PreTokenize implements PreTokenizer interface for BertPreTokenizer
func (bt *BertPreTokenizer) PreTokenize(pretokenized *tokenizer.PreTokenizedString) (*tokenizer.PreTokenizedString, error) {
	pretok := pretokenized.Split(func(noop int, sub *normalizer.NormalizedString) []tokenizer.SplitIdx {
		var splits []normalizer.NormalizedString
		whitespace := normalizer.NewRegexpPattern(`\s+`)
		wsSubs := sub.Split(whitespace, normalizer.RemovedBehavior)

		for _, sub := range wsSubs {
			puncSubs := sub.Split(normalizer.NewFnPattern(isBertPunc), normalizer.IsolatedBehavior)
			splits = append(splits, puncSubs...)
		}

		var splitIdxs []tokenizer.SplitIdx
		for _, s := range splits {
			normalized := s
			splitIdx := tokenizer.SplitIdx{Normalized: &normalized, Tokens: nil}
			splitIdxs = append(splitIdxs, splitIdx)
		}

		return splitIdxs
	})

	return pretok, nil
}
