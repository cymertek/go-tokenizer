package normalizer

import (
	"slices"
	"unicode"
)

// BertNormalizer normalizes text for BERT-style tokenization: lowercase, strip accents, handle Chinese characters.
type BertNormalizer struct {
	CleanText          bool `json:"clean_text"`           // CleanText removes control characters and collapses whitespace into single spaces
	Lowercase          bool `json:"lowercase"`            // Lowercase applies case folding to all text
	HandleChineseChars bool `json:"handle_chinese_chars"` // HandleChineseChars inserts spaces around CJK characters for splitting
	StripAccents       bool `json:"strip_accents"`        // StripAccents removes diacritical marks from characters
}

// NewBertNormalizer creates a new BERT normalizer with the given configuration flags.
func NewBertNormalizer(cleanText, lowercase, handleChineseChars, stripAccents bool) *BertNormalizer {
	return &BertNormalizer{
		CleanText:          cleanText,
		Lowercase:          lowercase,
		HandleChineseChars: handleChineseChars,
		StripAccents:       stripAccents,
	}
}

// IsWhitespace checks whether rune c is a BERT whitespace character
func isWhitespace(c rune) bool {
	// NOTE. `unicode.IsSpace(c rune)` has more cases
	switch c {
	case ' ', '\t', '\n', '\r':
		return true
	default:
		return false
	}
}

// IsControl checks whether rune c is a BERT control character
func isControl(c rune) bool {
	switch c {
	case '\t':
		return false
	case '\n':
		return false
	case '\r':
		return false
	}
	return unicode.In(c, unicode.Cc, unicode.Cf)
}

// bpunc is the BERT extension of the Punctuation character range
var bpunc = &unicode.RangeTable{
	R16: []unicode.Range16{
		{0x0021, 0x002f, 1}, // 33-47
		{0x003a, 0x0040, 1}, // 58-64
		{0x005b, 0x0060, 1}, // 91-96
		{0x007b, 0x007e, 1}, // 123-126
	},
	LatinOffset: 4, // All less than 0x00FF
}

// IsPunctuation checks whether rune c is a BERT punctuation character
func isPunctuation(c rune) bool {
	return unicode.In(c, bpunc, unicode.P)
}

// This defines a "chinese character" as anything in the CJK Unicode block:
//
//	https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
//
// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
// despite its name. The modern Korean Hangul alphabet is a different block,
// as is Japanese Hiragana and Katakana. Those alphabets are used to write
// space-separated words, so they are not treated specially and handled
// like for all of the other languages.
var cjk = &unicode.RangeTable{

	R16: []unicode.Range16{
		{0x4e00, 0x9fff, 1},
		{0x3400, 0x4dbf, 1},
		{0xf900, 0xfaff, 1},
	},
	R32: []unicode.Range32{
		{Lo: 0x20000, Hi: 0x2a6df, Stride: 1},
		{Lo: 0x2a700, Hi: 0x2b73f, Stride: 1},
		{Lo: 0x2b740, Hi: 0x2b81f, Stride: 1},
		{Lo: 0x2b820, Hi: 0x2ceaf, Stride: 1},
		{Lo: 0x2f800, Hi: 0x2fa1f, Stride: 1},
	},
}

// isChinese validates that rune c is in the CJK range according to BERT spec
func isChinese(c rune) bool {
	// 0x4E00..=0x9FFF => true,
	// 0x3400..=0x4DBF => true,
	// 0x20000..=0x2A6DF => true,
	// 0x2A700..=0x2B73F => true,
	// 0x2B740..=0x2B81F => true,
	// 0x2B920..=0x2CEAF => true,
	// 0xF900..=0xFAFF => true,
	// 0x2F800..=0x2FA1F => true,

	return unicode.In(c, cjk)

	// return unicode.Is(unicode.Han, c)
}

func doCleanText(n *NormalizedString) *NormalizedString {

	s := n.normalized
	var changeMap []ChangeMap

	// Fisrt, reverse the string
	oRunes := []rune(s)

	revRunes := make([]rune, 0, len(oRunes))
	for i := range slices.Backward(oRunes) {
		revRunes = append(revRunes, oRunes[i])
	}

	// Then, clean up
	removed := 0
	for _, r := range revRunes {
		if r == 0 || r == 0xfffd || isControl(r) {
			removed += 1
		} else {
			if removed > 0 {
				if isWhitespace(r) {
					r = ' '
				}
				changeMap = append(changeMap, ChangeMap{
					RuneVal: string(r),
					Changes: -removed,
				})
				removed = 0
			} else if removed == 0 {
				changeMap = append(changeMap, ChangeMap{
					RuneVal: string(r),
					Changes: 0,
				})
			}
		}
	}

	// Flip back changeMap (reverse the slice)
	unrevMap := make([]ChangeMap, len(changeMap))
	for i, v := range slices.Backward(changeMap) {
		unrevMap[len(changeMap)-1-i] = v
	}

	return n.Transform(unrevMap, removed)
}

func doHandleChineseChars(n *NormalizedString) *NormalizedString {
	var changeMap []ChangeMap
	for _, r := range n.normalized {
		// padding around chinese char
		if isChinese(r) {
			changeMap = append(changeMap, []ChangeMap{
				{
					RuneVal: string(' '),
					Changes: 1,
				},
				{
					RuneVal: string(r),
					Changes: 0,
				},
				{
					RuneVal: string(' '),
					Changes: 1,
				},
			}...)
		} else {
			changeMap = append(changeMap, ChangeMap{string(r), 0})
		}
	}

	return n.Transform(changeMap, 0)
}

func doLowercase(n *NormalizedString) *NormalizedString {
	return n.Lowercase()
}

func stripAccents(n *NormalizedString) *NormalizedString {
	return n.RemoveAccents()
}

// Normalize implements Normalizer interface for BertNormalizer
func (bn *BertNormalizer) Normalize(n *NormalizedString) (*NormalizedString, error) {
	if bn.CleanText {
		n = doCleanText(n)
	}

	if bn.HandleChineseChars {
		n = doHandleChineseChars(n)
	}

	if bn.Lowercase {
		n = doLowercase(n)
	}

	if bn.StripAccents {
		n = stripAccents(n)
	}

	return n, nil
}

// export some functions

// IsBertPunctuation checks whether an input rune is a BERT punctuation
func IsBertPunctuation(c rune) bool {
	return isPunctuation(c)
}

// IsBertWhitespace checks whether an input rune is a BERT whitespace
func IsBertWhitespace(c rune) bool {
	return isWhitespace(c)
}

// IsPunctuation returns whether input rune is a punctuation or not.
func IsPunctuation(c rune) bool {
	return isPunctuation(c)
}

// IsWhitespace checks whether an input rune is a whitespace
func IsWhitespace(c rune) bool {
	return isWhitespace(c)
}
