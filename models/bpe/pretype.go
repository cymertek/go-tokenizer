package bpe

import (
	"strings"
	"unicode/utf8"
)

// PreSplit represents a single pre-tokenized fragment from the state machine.
type PreSplit struct {
	Text string
}

// HasSpace reports whether this fragment starts with whitespace.
func (ps PreSplit) HasSpace() bool {
	if len(ps.Text) == 0 {
		return false
	}
	c := ps.Text[0]
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

// TrimSpace returns a copy with leading/trailing whitespace removed.
func (ps PreSplit) TrimSpace() PreSplit {
	return PreSplit{Text: strings.TrimSpace(ps.Text)}
}

// decodeRune decodes the next rune from text starting at position i, returning
// the rune, its byte width, and the new position. Uses utf8.DecodeRuneInString
// for proper multi-byte UTF-8 handling.
func decodeRune(text string, i int) (rune, int, int) {
	r, w := utf8.DecodeRuneInString(text[i:])
	return r, w, i + w
}

// kindOf returns the kind of a rune for pre-tokenization purposes.
func kindOf(r rune) kind {
	if r < 0x80 {
		switch {
		case r == ' ' || r == '\t':
			return kindSpace
		case r == '\n' || r == '\r':
			return kindNewline
		case 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z':
			return kindLetter
		case '0' <= r && r <= '9':
			return kindDigit
		default:
			return kindPunct
		}
	}
	return runeKind(r)
}

// advanceByRune advances position i by the width of the rune at that position.
func advanceByRune(text string, i int) int {
	_, w := utf8.DecodeRuneInString(text[i:])
	return i + w
}

// ---- Pre-tokenization state machine ----

type kind int

const (
	kindSpace kind = iota
	kindNewline
	kindLetter
	kindDigit
	kindMark
	kindCJK
	kindPunct
	kindOther
)

func runeKind(r rune) kind {
	if r < 0x80 {
		switch {
		case r == ' ' || r == '\t':
			return kindSpace
		case r == '\n' || r == '\r':
			return kindNewline
		case 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z':
			return kindLetter
		case '0' <= r && r <= '9':
			return kindDigit
		default:
			return kindPunct
		}
	}

	switch {
	case r >= 0x00C0 && r <= 0x024F: // Latin Extended-A/B
		return kindLetter
	case r >= 0x0370 && r <= 0x03FF: // Greek
		return kindLetter
	case r >= 0x0400 && r <= 0x04FF: // Cyrillic
		return kindLetter
	case r >= 0x0530 && r <= 0x055F: // Armenian
		return kindLetter
	case r >= 0x0600 && r <= 0x06FF: // Arabic
		return kindLetter
	case r >= 0x0900 && r <= 0x097F: // Devanagari
		return kindLetter
	case r >= 0x1F00 && r <= 0x1FFF: // Greek Extended
		return kindLetter
	case r >= 0x2C60 && r <= 0x2C7F: // Latin Extended-C
		return kindLetter
	case r >= 0x1E00 && r <= 0x1EFF: // Latin Extended Additional
		return kindLetter
	}

	switch {
	case r >= 0x0300 && r <= 0x036F: // Combining Diacritical Marks
		return kindMark
	case r >= 0x1DC0 && r <= 0x1DFF: // Combining Diacritical Marks Supplement
		return kindMark
	case r >= 0x20D0 && r <= 0x20FF: // Combining Diactical Marks for Symbols
		return kindMark
	case r >= 0xFE20 && r <= 0xFE2F: // Combining Half Marks
		return kindMark
	}

	switch {
	case r >= 0x4E00 && r <= 0x9FFF: // CJK Unified Ideographs
		return kindCJK
	case r >= 0x3400 && r <= 0x4DBF: // CJK Extension A
		return kindCJK
	case r >= 0xF900 && r <= 0xFAFF: // CJK Compatibility Ideographs
		return kindCJK
	case r >= 0x3000 && r <= 0x303F: // CJK Symbols and Punctuation
		return kindCJK
	case r >= 0x3040 && r <= 0x309F: // Hiragana
		return kindCJK
	case r >= 0x30A0 && r <= 0x30FF: // Katakana
		return kindCJK
	case r >= 0xAC00 && r <= 0xD7AF: // Hangul Syllables
		return kindCJK
	case r >= 0x1100 && r <= 0x11FF: // Hangul Jamo
		return kindCJK
	}

	switch {
	case r >= 0x0660 && r <= 0x0669: // Arabic-Indic
		return kindDigit
	case r >= 0x06F0 && r <= 0x06F9: // Extended Arabic-Indic
		return kindDigit
	case r >= 0x0966 && r <= 0x096F: // Devanagari
		return kindDigit
	case r >= 0x0E50 && r <= 0x0E59: // Thai
		return kindDigit
	}

	return kindOther
}

// stripWS strips trailing ASCII-whitespace from a whitespace fragment.
func stripWS(s string) string {
	n := len(s)
	for n > 0 {
		c := s[n-1]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			n--
		} else {
			break
		}
	}
	return s[:n]
}

func isSpace(b byte) bool  { return b == ' ' || b == '\t' }
func isNL(b byte) bool     { return b == '\n' || b == '\r' }
func isLetter(b byte) bool { return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') }
func isDigit(b byte) bool  { return b >= '0' && b <= '9' }

// modeQwen2 handles Qwen2, StableLM2: numbers-first, then contractions,
// then letters (with optional punct prefix), punct, newlines, whitespace.
func modeQwen2(text string) []PreSplit {
	n := len(text)
	var out []PreSplit
	i := 0

	for i < n {
		b := text[i]

		if isSpace(b) {
			start := i
			for i < n && isSpace(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if isNL(b) {
			start := i
			for i < n && isNL(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if isDigit(b) {
			start := i
			for i < n && isDigit(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if b == '\'' && i+1 < n {
			b2 := text[i+1]
			if i+2 < n {
				b3 := text[i+2]
				if (b2 == 'r' || b2 == 'R') && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'v' && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'l' && (b3 == 'l' || b3 == 'L') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
			}
			if (b2 == 's' || b2 == 'S') || (b2 == 't' || b2 == 'T') || (b2 == 'm' || b2 == 'M') || (b2 == 'd' || b2 == 'D') {
				out = append(out, PreSplit{Text: text[i : i+2]})
				i += 2
				continue
			}
		}

		if isLetter(b) {
			start := i
			for i < n && isLetter(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		start := i
		for i < n && !isSpace(text[i]) && !isNL(text[i]) && !isLetter(text[i]) && !isDigit(text[i]) {
			i++
		}
		if i > start {
			out = append(out, PreSplit{Text: text[start:i]})
		}
		continue
	}

	return out
}

// modeGPT2 handles GPT-2: contractions-first, then letters, then numbers, punct, whitespace.
func modeGPT2(text string) []PreSplit {
	n := len(text)
	var out []PreSplit
	i := 0

	for i < n {
		b := text[i]

		if isSpace(b) || b == '\n' || b == '\r' {
			start := i
			for i < n && (isSpace(text[i]) || text[i] == '\n' || text[i] == '\r') {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if b == '\'' && i+1 < n {
			b2 := text[i+1]
			if i+2 < n {
				b3 := text[i+2]
				if (b2 == 'r' || b2 == 'R') && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'v' && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'l' && (b3 == 'l' || b3 == 'L') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
			}
			if (b2 == 's' || b2 == 'S') || (b2 == 't' || b2 == 'T') || (b2 == 'm' || b2 == 'M') || (b2 == 'd' || b2 == 'D') {
				out = append(out, PreSplit{Text: text[i : i+2]})
				i += 2
				continue
			}
		}

		if isLetter(b) {
			start := i
			for i < n && isLetter(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if isDigit(b) {
			out = append(out, PreSplit{Text: text[i : i+1]})
			i++
			continue
		}

		if b < 0x80 {
			start := i
			for i < n && !isSpace(text[i]) && !isNL(text[i]) && !isLetter(text[i]) && !isDigit(text[i]) && text[i] < 0x80 {
				i++
			}
			if i > start {
				out = append(out, PreSplit{Text: text[start:i]})
			}
			continue
		}

		k := runeKind(rune(b))
		switch k {
		case kindMark:
			r, w, ni := decodeRune(text, i)
			_ = r
			out = append(out, PreSplit{Text: text[i : ni]})
			i = ni
			_ = w
		case kindCJK:
			start := i
			for i < n {
				r, _, ni := decodeRune(text, i)
				if !isCJK(r) {
					break
				}
				i = ni
			}
			out = append(out, PreSplit{Text: text[start:i]})
		case kindNewline:
			start := i
			for i < n && isNL(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
		case kindPunct, kindOther:
			start := i
			for i < n {
				r, w, ni := decodeRune(text, i)
				if runeKind(r) != k {
					break
				}
				i = ni
				_ = w
			}
			if i > start {
				out = append(out, PreSplit{Text: text[start:i]})
			}
		default:
			r, _, ni := decodeRune(text, i)
			_ = r
			out = append(out, PreSplit{Text: text[i : ni]})
			i = ni
		}
	}

	return out
}

// modeLlama3 handles Llama3: numbers max 3 digits.
func modeLlama3(text string) []PreSplit {
	n := len(text)
	var out []PreSplit
	i := 0

	for i < n {
		b := text[i]

		if isSpace(b) {
			start := i
			for i < n && isSpace(text[i]) {
				i++
			}
			frag := stripWS(text[start:i])
			if frag != "" {
				out = append(out, PreSplit{Text: frag})
			}
			continue
		}

		if isNL(b) {
			start := i
			for i < n && isNL(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if isDigit(b) {
			start := i
			count := 0
			for i < n && isDigit(text[i]) && count < 3 {
				i++
				count++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if b == '\'' && i+1 < n {
			b2 := text[i+1]
			if i+2 < n {
				b3 := text[i+2]
				if (b2 == 'r' || b2 == 'R') && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'v' && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'l' && (b3 == 'l' || b3 == 'L') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
			}
			if (b2 == 's' || b2 == 'S') || (b2 == 't' || b2 == 'T') || (b2 == 'm' || b2 == 'M') || (b2 == 'd' || b2 == 'D') {
				out = append(out, PreSplit{Text: text[i : i+2]})
				i += 2
				continue
			}
		}

		if isLetter(b) {
			start := i
			for i < n && isLetter(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		start := i
		for i < n && !isSpace(text[i]) && !isNL(text[i]) && !isLetter(text[i]) && !isDigit(text[i]) {
			i++
		}
		if i > start {
			out = append(out, PreSplit{Text: text[start:i]})
		}
		continue
	}

	return out
}

// modeQwen35 handles Qwen35: diacritics first, then numbers, then contractions.
func modeQwen35(text string) []PreSplit {
	n := len(text)
	var out []PreSplit
	i := 0

	for i < n {
		b := text[i]

		if isSpace(b) {
			start := i
			for i < n && isSpace(text[i]) {
				i++
			}
			frag := stripWS(text[start:i])
			if frag != "" {
				out = append(out, PreSplit{Text: frag})
			}
			continue
		}

		if isNL(b) {
			start := i
			for i < n && isNL(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		k := runeKind(rune(b))
		if k == kindMark {
			start := i
			for i < n && runeKind(rune(text[i])) == kindMark {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if isDigit(b) {
			start := i
			for i < n && isDigit(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if b == '\'' && i+1 < n {
			b2 := text[i+1]
			if i+2 < n {
				b3 := text[i+2]
				if (b2 == 'r' || b2 == 'R') && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'v' && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'l' && (b3 == 'l' || b3 == 'L') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
			}
			if (b2 == 's' || b2 == 'S') || (b2 == 't' || b2 == 'T') || (b2 == 'm' || b2 == 'M') || (b2 == 'd' || b2 == 'D') {
				out = append(out, PreSplit{Text: text[i : i+2]})
				i += 2
				continue
			}
		}

		if isLetter(b) {
			start := i
			for i < n && isLetter(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		if k == kindPunct || k == kindOther {
			start := i
			for i < n && (runeKind(rune(text[i])) == kindPunct || runeKind(rune(text[i])) == kindOther) {
				i++
			}
			if i > start {
				out = append(out, PreSplit{Text: text[start:i]})
			}
			continue
		}

		if k == kindCJK {
			start := i
			for i < n && isCJK(rune(text[i])) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}

		out = append(out, PreSplit{Text: text[i : i+1]})
		i++
	}

	return out
}

func modeGPT4O(text string) []PreSplit { return modeLlama3(text) }

// ---- Sequential rule functions ----

func ruleNewlines(text string) []PreSplit {
	i := 0
	n := len(text)
	var matches []PreSplit
	for i < n {
		if text[i] == '\n' || text[i] == '\r' {
			start := i
			for i < n && (text[i] == '\n' || text[i] == '\r') {
				i++
			}
			matches = append(matches, PreSplit{Text: text[start:i]})
		} else {
			i++
		}
	}
	return matches
}

func ruleSingleDigit(text string) []PreSplit {
	i := 0
	n := len(text)
	var result []PreSplit
	for i < n {
		if isDigit(text[i]) {
			result = append(result, PreSplit{Text: text[i : i+1]})
			i++
		} else {
			start := i
			for i < n && !isDigit(text[i]) {
				i++
			}
			if i > start {
				result = append(result, PreSplit{Text: text[start:i]})
			}
		}
	}
	return result
}

func rulePunctuation(text string) []PreSplit {
	i := 0
	n := len(text)
	var result []PreSplit
	for i < n {
		b := text[i]
		if !isSpace(b) && !isNL(b) && !isLetter(b) && !isDigit(b) && b < 0x80 {
			start := i
			for i < n && !isSpace(text[i]) && !isNL(text[i]) && !isLetter(text[i]) && !isDigit(text[i]) && text[i] < 0x80 {
				i++
			}
			result = append(result, PreSplit{Text: text[start:i]})
		} else {
			start := i
			for i < n && (isSpace(text[i]) || isNL(text[i]) || isLetter(text[i]) || isDigit(text[i]) || text[i] >= 0x80) {
				i++
			}
			if i > start {
				result = append(result, PreSplit{Text: text[start:i]})
			}
		}
	}
	return result
}

func ruleCJK(text string) []PreSplit {
	i := 0
	n := len(text)
	var matches []PreSplit
	for i < n {
		if isCJK(rune(text[i])) {
			start := i
			for i < n && isCJK(rune(text[i])) {
				i++
			}
			matches = append(matches, PreSplit{Text: text[start:i]})
		} else {
			i++
		}
	}
	return matches
}

func isCJK(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF:
		return true
	case r >= 0x3400 && r <= 0x4DBF:
		return true
	case r >= 0xF900 && r <= 0xFAFF:
		return true
	case r >= 0x3000 && r <= 0x303F:
		return true
	case r >= 0x3040 && r <= 0x309F:
		return true
	case r >= 0x30A0 && r <= 0x30FF:
		return true
	case r >= 0xAC00 && r <= 0xD7AF:
		return true
	case r >= 0x1100 && r <= 0x11FF:
		return true
	}
	return false
}

func composeRules(rules []func(string) []PreSplit) func(string) []PreSplit {
	return func(text string) []PreSplit {
		fragments := []PreSplit{{Text: text}}
		for _, rule := range rules {
			next := make([]PreSplit, 0, len(fragments)*2)
			for _, frag := range fragments {
				matches := rule(frag.Text)
				if len(matches) == 0 {
					next = append(next, frag)
					continue
				}
				next = append(next, matches...)
			}
			fragments = next
		}
		result := make([]PreSplit, 0, len(fragments))
		for _, f := range fragments {
			if f.Text != "" {
				result = append(result, f)
			}
		}
		return result
	}
}

// preTokenize is the main dispatcher for BPE pre-tokenization strategies.
func preTokenize(text string, pt PreType, spaceChar rune) []PreSplit {
	if text == "" {
		return nil
	}
	if pt == PreGemma4 {
		return []PreSplit{{Text: text}}
	}

	var raw []PreSplit
	switch pt {
	case PreQwen2, PreStableLM2:
		raw = modeQwen2(text)
	case PreGPT2:
		raw = modeGPT2(text)
	case PreLlama3:
		raw = modeLlama3(text)
	case PreQwen35:
		raw = modeQwen35(text)
	case PreGPT4O:
		raw = modeGPT4O(text)
	case PreStarcoder:
		raw = preStarCoder(text)
	case PreDeepSeekLLM:
		raw = composeRules([]func(string) []PreSplit{
			ruleNewlines,
			func(s string) []PreSplit {
				var matches []PreSplit
				i := 0
				n := len(s)
				start := -1
				for i < n {
					r := rune(s[i])
					k := runeKind(r)
					if k == kindLetter || (k == kindSpace && start == -1) {
						if start == -1 && k == kindSpace {
							start = i
						} else if start < 0 {
							start = i
						}
						i++
					} else {
						if start >= 0 {
							matches = append(matches, PreSplit{Text: s[start:i]})
							start = -1
						}
						i++
					}
				}
				if start >= 0 {
					matches = append(matches, PreSplit{Text: s[start:i]})
				}
				return matches
			},
			rulePunctuation, ruleCJK, ruleSingleDigit,
		})(text)
	case PreFalcon:
		raw = preFalcon(text)
	case PreDefault:
		raw = composeRules([]func(string) []PreSplit{
			rulePunctuation,
			func(s string) []PreSplit { return modeGPT2(s) },
			ruleSingleDigit, ruleSingleDigit,
		})(text)
	default:
		raw = modeQwen2(text)
	}

	return normalizeFragments(raw, spaceChar)
}

func preStarCoder(text string) []PreSplit {
	var frags []PreSplit
	i := 0
	n := len(text)
	if i < n && isDigit(text[i]) {
		start := i
		for i < n && isDigit(text[i]) {
			i++
		}
		frags = append(frags, PreSplit{Text: text[start:i]})
	}
	if i < n {
		rest := modeGPT2(text[i:])
		for _, f := range rest {
			b := f.Text[0]
			if b == '\'' {
				frags = append(frags, f)
				continue
			}
			if !isSpace(b) && !isNL(b) && !isLetter(b) && !isDigit(b) && b < 0x80 {
				continue
			}
			frags = append(frags, f)
		}
	}
	return frags
}

func preFalcon(text string) []PreSplit {
	frags := splitDigitsIntoThree(text)
	var result []PreSplit
	for _, f := range frags {
		puncts := rulePunctuation(f.Text)
		if puncts == nil {
			result = append(result, f)
		} else {
			for _, p := range puncts {
				result = append(result, modeGPT2PreserveDigits(p.Text)...)
			}
		}
	}
	return result
}

func modeGPT2PreserveDigits(text string) []PreSplit {
	n := len(text)
	var out []PreSplit
	i := 0
	for i < n {
		b := text[i]
		if isSpace(b) {
			start := i
			for i < n && isSpace(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}
		if isNL(b) {
			start := i
			for i < n && isNL(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}
		if b == '\'' && i+1 < n {
			b2 := text[i+1]
			if i+2 < n {
				b3 := text[i+2]
				if (b2 == 'r' || b2 == 'R') && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'v' && (b3 == 'e' || b3 == 'E') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
				if b2 == 'l' && (b3 == 'l' || b3 == 'L') {
					out = append(out, PreSplit{Text: text[i : i+3]})
					i += 3
					continue
				}
			}
			if (b2 == 's' || b2 == 'S') || (b2 == 't' || b2 == 'T') || (b2 == 'm' || b2 == 'M') || (b2 == 'd' || b2 == 'D') {
				out = append(out, PreSplit{Text: text[i : i+2]})
				i += 2
				continue
			}
		}
		if isLetter(b) {
			start := i
			for i < n && isLetter(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}
		if isDigit(b) {
			start := i
			for i < n && isDigit(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
			continue
		}
		if b < 0x80 {
			start := i
			for i < n && !isSpace(text[i]) && !isNL(text[i]) && !isLetter(text[i]) && !isDigit(text[i]) && text[i] < 0x80 {
				i++
			}
			if i > start {
				out = append(out, PreSplit{Text: text[start:i]})
			}
			continue
		}
		k := runeKind(rune(b))
		switch k {
		case kindMark:
			out = append(out, PreSplit{Text: text[i : i+1]})
			i++
		case kindCJK:
			start := i
			for i < n && isCJK(rune(text[i])) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
		case kindNewline:
			start := i
			for i < n && isNL(text[i]) {
				i++
			}
			out = append(out, PreSplit{Text: text[start:i]})
		case kindPunct, kindOther:
			start := i
			for i < n && runeKind(rune(text[i])) == k {
				i++
			}
			if i > start {
				out = append(out, PreSplit{Text: text[start:i]})
			}
		default:
			out = append(out, PreSplit{Text: text[i : i+1]})
			i++
		}
	}
	return out
}

func splitDigitsIntoThree(text string) []PreSplit {
	var frags []PreSplit
	i := 0
	n := len(text)
	for i < n {
		if !isDigit(text[i]) {
			start := i
			for i < n && !isDigit(text[i]) {
				i++
			}
			frags = append(frags, PreSplit{Text: text[start:i]})
			continue
		}
		start := i
		for i < n && isDigit(text[i]) {
			i++
		}
		run := text[start:i]
		for len(run) >= 3 {
			frags = append(frags, PreSplit{Text: run[:3]})
			run = run[3:]
		}
		if len(run) > 0 {
			frags = append(frags, PreSplit{Text: run})
		}
	}
	return frags
}

// normalizeFragments converts leading ASCII whitespace to the vocab's space prefix
// character and merges pure-whitespace fragments into the next content fragment.
func normalizeFragments(frags []PreSplit, spaceChar rune) []PreSplit {
	if spaceChar == 0 {
		var out []PreSplit
		for _, f := range frags {
			if !isPureSpace(f.Text) {
				out = append(out, f)
			}
		}
		return out
	}

	spaceStr := string(spaceChar)
	var out []PreSplit
	var pendingSpaces int

	for _, f := range frags {
		if isPureSpace(f.Text) {
			for i := 0; i < len(f.Text); i++ {
				c := f.Text[i]
				if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
					pendingSpaces++
				}
			}
			continue
		}

		text := f.Text
		contentSpaces := 0
		for i := 0; i < len(text); i++ {
			c := text[i]
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				contentSpaces++
			} else {
				break
			}
		}
		totalSpaces := pendingSpaces + contentSpaces

		if totalSpaces > 0 {
			text = text[contentSpaces:]
			text = spaceStr + text
			rem := totalSpaces - 1
			if rem > 0 {
				out = append(out, PreSplit{Text: strings.Repeat(spaceStr, rem)})
			}
			pendingSpaces = 0
		}
		out = append(out, PreSplit{Text: text})
	}

	// Output any remaining pending spaces as separate tokens.
	if pendingSpaces > 0 {
		for i := 0; i < pendingSpaces; i++ {
			out = append(out, PreSplit{Text: spaceStr})
		}
	}

	return out
}

func isPureSpace(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return false
		}
	}
	return len(s) > 0
}
