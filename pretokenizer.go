package tokenizer

// wrapper for subpart of NormalizedString

import (
	"fmt"
	"log"
	// "reflect"

	"github.com/cymertek/go-tokenizer/normalizer"
)

// PreToken represents a pre-tokenized segment with its value string, byte offsets, and optional tokens.
type PreToken struct {
	Value   string
	Offsets []int
	Tokens  []Token // optional
}

// OffsetType specifies whether offsets are in bytes or characters.
type OffsetType int

const (
	// Byte indicates byte-level offsets.
	Byte OffsetType = iota
	// Char indicates character-level offsets.
	Char
)

// Split contains the underlying `NormalizedString` as well as
// its offsets in the original string. These offsets are in the
// `original` referential. It also contains any `Token` associated
// to the current split
type Split struct {
	// OriginalOffset is the start offset of this split in the original string.
	originalOffset int

	// Normalized is the underlying `NormalizedString`. Each SubString
	// is represented by a `NormalizedString`. In the end, there might
	// be many SubStrings representing various parts of the original
	// input string.
	normalized *normalizer.NormalizedString

	// Optional Tokens associated with this split
	tokens []Token
}

// NewSplit creates a new Split from a input NormalizedString
func NewSplit(normalized *normalizer.NormalizedString, tokens []Token) Split {
	return Split{normalized: normalized, tokens: tokens}
}

// PreTokenizedString manages pre-tokenization of a string into splits with offset tracking.
type PreTokenizedString struct {
	original string
	splits   []Split
}

// SplitFn takes a `NormalizedString` and returns an iterator over the
// produced `NormalizedString`.
//
// NOTE. SplitFn is free of modifying these `NormalizedString` as long as:
// The produced `NormalizedString`, if combined back together, must have
// the same `original` string as the original one given to `SplitFn`. This
// means that for the offsets tracking to work as expected, `SplitFn` must
// produce "splits" of the ORIGINAL string.
type SplitFn func(int, *normalizer.NormalizedString) []SplitIdx

// Split splits the `PreTokenizedString` by providing a `SplitFn` which is in
// charge of splitting each substring (`NormalizedString`) into multiple parts.
// func (pt *PreTokenizedString) Split(splitFn SplitFn) *PreTokenizedString {
func (pt *PreTokenizedString) Split(splitFn SplitFn) *PreTokenizedString {

	var newSplits []Split
	for i, originalSplit := range pt.splits {
		if originalSplit.tokens != nil {
			newSplits = append(newSplits, originalSplit)
			continue
		}

		// Compute the original offset for this split by accumulating
		// the lengths of previously created pieces.
		origOff := 0
		for _, s := range newSplits {
			origOff += len(s.normalized.GetOriginal())
		}

		for _, splitIdx := range splitFn(i, originalSplit.normalized) {
			if splitIdx.Normalized != nil && splitIdx.Normalized.GetNormalized() != "" {
				// split := NewSplit(splitIdx.Normalized, splitIdx.Tokens)
				split := Split{
					originalOffset: origOff,
					normalized:     splitIdx.Normalized,
					tokens:         splitIdx.Tokens,
				}
				newSplits = append(newSplits, split)
				origOff += len(splitIdx.Normalized.GetOriginal())
			}
		}
	}

	pt.splits = newSplits
	return pt
}

// Normalize normalizes all the splits that do not have attached `Tokens`,
// using the provided `normalize` function.
func (pt *PreTokenizedString) Normalize(nFn func(*normalizer.NormalizedString) *normalizer.NormalizedString) *PreTokenizedString {

	var nSplits []Split

	for _, split := range pt.splits {
		if split.tokens == nil {
			newSplit := split
			newSplit.normalized = nFn(split.normalized)
			nSplits = append(nSplits, newSplit)
		}
	}

	pt.splits = nSplits
	return pt
}

// Tokenize tokenizes all the splits that do not have attached `Tokens`, using the provided
// `tokenize` function
func (pt *PreTokenizedString) Tokenize(tokFn func(*normalizer.NormalizedString) ([]Token, error)) (*PreTokenizedString, error) {
	var nSplits []Split

	for _, split := range pt.splits {
		newSplit := split
		if split.tokens == nil {
			toks, err := tokFn(split.normalized)
			if err != nil {
				return nil, err
			}
			newSplit.tokens = toks
		}
		nSplits = append(nSplits, newSplit)
	}

	pt.splits = nSplits
	return pt, nil
}

// OriginalOffset returns the start offset in the original string for the split at the given index.
func (pt *PreTokenizedString) OriginalOffset(idx int) int {
	offset := 0
	for i := 0; i < idx && i < len(pt.splits); i++ {
		offset += len(pt.splits[i].normalized.GetOriginal())
	}
	return offset
}

// IntoEncoding transforms the current `PreTokenizedString` into an `Encoding`.
//
// If a `wordIdx` is provided, any word in the generated `Encoding`
// will be set to this value. This is generally used with pre-tokenized
// input, that do not need the `PreTokenizedString` to generate word ids.
//
// This method will fail if some splits do not have associated `Token`.
func (pt *PreTokenizedString) IntoEncoding(typeID int, wordIdx int, offsetType OffsetType) (*Encoding, error) {

	if len(pt.splits) == 0 {
		return DefaultEncoding(), nil
	}

	for _, s := range pt.splits {
		if len(s.tokens) == 0 {
			err := fmt.Errorf("split has not been tokenized. call PreTokenizeString.Tokenize() method first")
			return nil, err
		}
	}

	charMap := make(map[int]int, 0) // map[byteIdx]runeIdx

	switch offsetType {
	case Char:
		currRuneIdx := 0
		for byteIdx, r := range pt.original {
			n := 0
			for i := 0; i < len([]byte(string(r))); i++ {
				charMap[byteIdx+n] = currRuneIdx
				n++
			}
			currRuneIdx++
		}
	case Byte:
		charMap = make(map[int]int, 0)

	default:
		err := fmt.Errorf("invalid offset type %v", offsetType)
		return nil, err
	}

	var (
		enIds               []int
		enTokens            []string
		enWords             []int
		enTypeIds           []int
		enOffsets           [][]int
		enSpecialTokensMask []int
		enAttentionMask     []int
	)

	for idx, split := range pt.splits {
		normalized := split.normalized
		offsets := normalized.OffsetsOriginal()
		charMapSplit := charMap
		var convertedOffsets []int
		for _, tok := range split.tokens {
			o := normalized.ConvertOffset(normalizer.NewRange(tok.Offsets[0], tok.Offsets[1], normalizer.NormalizedTarget))
			if o == nil {
				convertedOffsets = []int{offsets[0] + tok.Offsets[0], offsets[0] + tok.Offsets[1]}
			} else {
				convertedOffsets = []int{offsets[0] + o.Start(), offsets[0] + o.End()}
			}

			// Convert to char offset if relevant
			start, ok := charMapSplit[convertedOffsets[0]]
			if !ok {
				start = -1
			}
			end, ok := charMapSplit[convertedOffsets[1]]
			if !ok {
				end = -1
			}

			var newConvertedOffsets []int
			switch {
			case start != -1 && end != -1:
				newConvertedOffsets = []int{start, end}
			case start != -1 && end == -1: // If we reached the end, `end` is not in the map
				// But the one just before should be
				last, ok := charMapSplit[convertedOffsets[1]-1]
				if !ok {
					log.Printf("Something wrong here. Should find from map.\n")
					last = start + 1
				}
				newConvertedOffsets = []int{start, last}

			default:
				newConvertedOffsets = convertedOffsets
			}

			wordIndex := wordIdx
			if wordIdx == -1 {
				wordIndex = idx
			}

			// fmt.Printf("tok: ....%v - value: '%v'\n", tok, tok.Value)
			// NOTE: we get token value from offsets on normalized.

			enIds = append(enIds, tok.Id)
			enTokens = append(enTokens, tok.Value)
			enOffsets = append(enOffsets, newConvertedOffsets)
			enWords = append(enWords, wordIndex)
			enTypeIds = append(enTypeIds, typeID)
			enSpecialTokensMask = append(enSpecialTokensMask, 0)
			enAttentionMask = append(enAttentionMask, 1)
		}
	}

	en := DefaultEncoding()
	en.Ids = enIds
	en.Tokens = enTokens
	en.Offsets = enOffsets
	en.Words = enWords
	en.TypeIds = enTypeIds
	en.SpecialTokenMask = enSpecialTokensMask
	en.AttentionMask = enAttentionMask

	return en, nil
}

// GetSplits returns a list of splits, each of them being a slice of the normalized
// string, the associated offsets either in original or normalized
// referential, as well as the potention tokens
func (pt *PreTokenizedString) GetSplits(offsetRef normalizer.IndexOn, offsetType OffsetType) []PreToken {
	var preToks []PreToken

	var offsetConverter OffsetConverter
	if offsetType == Char {
		offsetConverter = NewBytesToCharOffsetConverter(pt.original)
	}

	offset := 0
	for _, s := range pt.splits {
		var offsets []int
		switch offsetRef {
		case normalizer.OriginalTarget:
			offsets = s.normalized.OffsetsOriginal()
		case normalizer.NormalizedTarget:
			length := s.normalized.Len()
			offset += length
			offsets = []int{offset - length, offset}
		}

		// Convert to char offsets if relevant
		if offsetConverter != nil {
			var err error
			offsets, err = offsetConverter.Convert(offsets)
			if err != nil {
				panic(err)
			}
		}

		preToks = append(preToks, PreToken{s.normalized.GetNormalized(), offsets, s.tokens})
	}

	return preToks
}

// NewPreTokenizedStringFromNS creates a PreTokenizedString from an existing NormalizedString, preserving its original text and splits.
func NewPreTokenizedStringFromNS(n *normalizer.NormalizedString) *PreTokenizedString {

	return &PreTokenizedString{
		original: n.GetOriginal(),
		splits:   []Split{{normalized: n, tokens: nil}},
	}
}

// NewPreTokenizedString create a new PreTokenizedString from input string
func NewPreTokenizedString(s string) *PreTokenizedString {
	n := normalizer.NewNormalizedFrom(s)
	return NewPreTokenizedStringFromNS(n)
}

type OffsetConverter interface {
	Convert(offsets []int) ([]int, error)
}

// BytesToCharOffsetConverter converts byte offsets to character (rune) offsets for UTF-8 strings.
type BytesToCharOffsetConverter struct {
	b2c map[int]int // map of byteIndex to character(rune) index
}

// NewBytesToCharOffsetConverter creates a converter that maps byte positions to rune positions in the given sequence.
func NewBytesToCharOffsetConverter(sequence string) *BytesToCharOffsetConverter {
	chars := []rune(sequence) // utf-8

	b2c := make(map[int]int)
	n := 0
	for charIdx, char := range chars {
		nbytes := len([]byte(string(char)))
		for i := 0; i < nbytes; i++ {
			byteIdx := n + i
			b2c[byteIdx] = charIdx
		}

		n += nbytes
	}

	return &BytesToCharOffsetConverter{b2c}
}

// Convert converts byte-indexed offsets to character-index offsets.
func (c *BytesToCharOffsetConverter) Convert(offsets []int) ([]int, error) {
	start, ok := c.b2c[offsets[0]]
	if !ok {
		err := fmt.Errorf("invalid offsets start %v", offsets[0])
		return nil, err
	}

	end, ok := c.b2c[offsets[1]]
	if !ok {
		err := fmt.Errorf("invalid offsets end %v", offsets[1])
		return nil, err
	}

	return []int{start, end}, nil
}
