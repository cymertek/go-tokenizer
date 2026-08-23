// Package bpe provides a trie-optimized BPE (Byte Pair Encoding) tokenizer.
// It uses greedy longest-match for fully-merged GGUF vocabularies, with
// pre-tokenization strategies matching llama.cpp's behavior.
package bpe

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/cymertek/go-tokenizer/models/common"
)

const ConcurrentThreshold = 32 * 1024 // 32KB — switch to concurrent encoding above this

// PreType is a local alias for the common.PreType enum used by pre-tokenization.
type PreType = common.PreType

const (
	// PreDefault uses the default pre-tokenization strategy (GPT-2-like with digit preservation).
	PreDefault = common.PreDefault
	// PreGPT2 uses GPT-2's contractions-first, letters, numbers, punctuation strategy.
	PreGPT2 = common.PreGPT2
	// PreQwen2 uses Qwen2's numbers-first, then contractions, then letters strategy.
	PreQwen2 = common.PreQwen2
	// PreLlama3 uses Llama 3's max-3-digit numbers strategy.
	PreLlama3 = common.PreLlama3
	// PreStarcoder uses StarCoder's digit-first with GPT-2 fallback strategy.
	PreStarcoder = common.PreStarcoder
	// PreDeepSeekLLM uses DeepSeek-LLM's multi-rule composition strategy.
	PreDeepSeekLLM = common.PreDeepSeekLLM
	// PreFalcon uses Falcon's digit-splitting with GPT-2 preservation strategy.
	PreFalcon = common.PreFalcon
	// PreQwen35 uses Qwen 3.5's diacritics-first, then numbers, then contractions strategy.
	PreQwen35 = common.PreQwen35
	// PreStableLM2 uses StableLM2's Qwen2-compatible pre-tokenization.
	PreStableLM2 = common.PreStableLM2
	// PreGPT4O uses GPT-4o's Llama 3-compatible pre-tokenization.
	PreGPT4O = common.PreGPT4O
	// PreGemma4 uses Gemma 4's no-pre-tokenization strategy (single fragment).
	PreGemma4 = common.PreGemma4
)

// BPE is a trie-optimized BPE tokenizer for GGUF files.
type BPE struct {
	vocab       map[string]int   // token string → id
	invVocab    map[int]string   // id → token string
	tokenType   []int32          // per-token type flags (parallel to vocab keys)
	trie        *byteTrie        // byte-level trie for O(k) longest-match
	bosID       int64            // beginning-of-sequence token ID (-1 if unset)
	eosID       int64            // end-of-sentence token ID (-1 if unset)
	eotID       int64            // end-of-text token ID (-1 if unset)
	eomID       int64            // end-of-message token ID (-1 if unset)
	unkID       int64            // unknown token ID (-1 if unset)
	preType     PreType          // pre-tokenization strategy
	addBOS      bool             // whether to prepend BOS token
	addEOS      bool             // whether to append EOS token
	spaceChar   rune             // space prefix character (e.g. Ġ = U+0120 for BPE models)
	cache       map[string][]int // optional encoding cache
	unkDisabled bool             // true when HasUNKID=true and UNKID=-1 (explicitly disabled)
}

// New creates a new BPE tokenizer from the given TokenizerData. The data must have
// Model == "bpe" with Tokens and optionally Merges set. Returns nil-safe: if data is
// nil, returns an empty BPE with default zero values for all fields.
func New(data *common.TokenizerData) (*BPE, error) {
	b := &BPE{
		vocab:    make(map[string]int),
		invVocab: make(map[int]string),
		bosID:    -1,
		eosID:    -1,
		unkID:    -1,
	}

	if data == nil {
		return b, nil
	}

	if len(data.Tokens) > 0 {
		b.trie = newByteTrie(data.Tokens)

		// Auto-detect spaceChar from vocabulary tokens.
		// BPE models typically use Ġ (U+0120) as a word boundary marker.
		for _, tok := range data.Tokens {
			if len(tok) > 0 {
				r, _ := utf8.DecodeRuneInString(tok)
				if r == 'Ġ' || r == '▁' || r == '_' || (r >= 0x0100 && r <= 0x017F) {
					b.spaceChar = r
					break
				}
			}
		}
	}
	for i, tok := range data.Tokens {
		b.vocab[tok] = i
		b.invVocab[i] = tok
	}

	// Preserve TokenType array if present (for GGUF compatibility).
	if len(data.TokenType) > 0 && len(data.TokenType) == len(b.invVocab) {
		b.tokenType = data.TokenType
	}

	if data.HasBOSID && data.BOSID >= 0 {
		b.bosID = data.BOSID
	}
	if data.HasEOSID && data.EOSID >= 0 {
		b.eosID = data.EOSID
	}
	if data.HasEOTID && data.EOTID >= 0 {
		b.eotID = data.EOTID
	}
	if data.HasEOMID && data.EOMID >= 0 {
		b.eomID = data.EOMID
	}
	if data.HasUNKID {
		if data.UNKID >= 0 {
			b.unkID = data.UNKID
		} else if b.unkID < 0 {
			// HasUNKID=true but UNKID=-1 means explicitly disabled (no fallback)
			b.unkDisabled = true
		}
	}
	if data.AddBOS {
		b.addBOS = true
	}
	if data.AddEOS {
		b.addEOS = true
	}
	if data.PreType != common.PreDefault {
		b.preType = PreType(data.PreType)
	} else if strings.Contains(strings.ToLower(data.Model), "gemma") || strings.Contains(strings.ToLower(data.Model), "gemini") {
		b.preType = PreGemma4
	}

	return b, nil
}

// Register registers the BPE tokenizer with the auto-registration registry.
func init() {
	common.Register("bpe", func(data *common.TokenizerData) (common.Tokenizer, error) {
		return New(data)
	})
}

// EncodeIDs tokenizes text into a slice of integer token IDs using greedy longest-match.
// If AddBOS/AddEOS are configured in the source data, BOS/EOS tokens are prepended/appended.
// Returns nil for empty input or nil receiver. Results may be cached if SetCache was called.
func (b *BPE) EncodeIDs(text string) []int {
	if b == nil || len(text) == 0 {
		return nil
	}

	if b.cache != nil {
		if ids, ok := b.cache[text]; ok {
			return ids
		}
	}

	var ids []int
	if len(text) > ConcurrentThreshold {
		ids = b.encodeIDsConcurrent(text)
	} else {
		ids = b.encodeIDsInner(text)
	}

	if b.cache != nil {
		b.cache[text] = ids
	}

	return ids
}

// encodeIDsConcurrent handles concurrent encoding for large texts.
func (b *BPE) encodeIDsConcurrent(text string) []int {
	ids := make([]int, 0, 256)
	if b.addBOS && b.bosID >= 0 {
		ids = append(ids, int(b.bosID))
	}
	ids = append(ids, b.encodeConcurrent(text)...)
	if b.addEOS && b.eosID >= 0 {
		ids = append(ids, int(b.eosID))
	}
	return ids
}

func (b *BPE) encodeIDsInner(text string) []int {
	ids := make([]int, 0, 16)

	if b.addBOS && b.bosID >= 0 {
		ids = append(ids, int(b.bosID))
	}

	ids = append(ids, b.tokenize(text)...)

	if b.addEOS && b.eosID >= 0 {
		ids = append(ids, int(b.eosID))
	}

	return ids
}

// tokenize pre-tokenizes the text, then encodes each fragment via the trie.
func (b *BPE) tokenize(text string) []int {
	if id, ok := b.vocab[text]; ok {
		return []int{id}
	}

	ids := make([]int, 0, len(text)/2+2) // heuristic: ~2 chars per token

	// First pass: identify all special token positions in the text.
	type specialSpan struct {
		start, end int // byte offsets in text
		id         int
	}
	var specials []specialSpan
	i := 0
	for i < len(text) {
		specialID, matchedLen := b.matchSpecialToken(text[i:])
		if matchedLen > 0 {
			specials = append(specials, specialSpan{i, i + matchedLen, specialID})
			i += matchedLen
			continue
		}
		i++
	}

	// Second pass: encode text, handling special spans and pre-tokenizing the gaps.
	pos := 0
	for _, sp := range specials {
		// Encode any gap before this special token.
		if sp.start > pos {
			gapIDs := b.encodeGap(text[pos:sp.start])
			ids = append(ids, gapIDs...)
		}
		ids = append(ids, sp.id)
		pos = sp.end
	}
	// Encode any trailing gap.
	if pos < len(text) {
		gapIDs := b.encodeGap(text[pos:])
		ids = append(ids, gapIDs...)
	}

	return ids
}

// encodeGap pre-tokenizes and encodes a text segment that contains no special tokens.
func (b *BPE) encodeGap(text string) []int {
	if len(text) == 0 {
		return nil
	}
	splits := preTokenize(text, b.preType, b.spaceChar)
	var ids []int
	for _, s := range splits {
		fragIDs := b.encodeFragment(s.Text)
		if len(fragIDs) == 0 && s.Text != "" {
			if !b.unkDisabled {
				fragIDs = b.encodeCharacters(s.Text)
			}
		}
		ids = append(ids, fragIDs...)
	}
	return ids
}

// matchSpecialToken checks if text[i:] starts with any multi-character "special" token
// from the vocabulary (e.g., <s>, </s>, [INST], ##word). Returns the token ID and
// length matched, or (0, 0) if no special token found.
func (b *BPE) matchSpecialToken(text string) (int, int) {
	if len(text) == 0 {
		return 0, 0
	}

	firstByte := text[0]

	// Only consider tokens that start with characters commonly used for special tokens:
	// < [ # (and ## for wordpiece continuations). This avoids matching arbitrary
	// punctuation sequences like ")<" or "*(" which should be pre-tokenized normally.
	isSpecialStart := firstByte == '<' || firstByte == '[' || firstByte == '#'

	if !isSpecialStart {
		return 0, 0
	}

	// Build a set of "special" tokens: multi-char tokens that start with punctuation
	// or are continuation markers (##). These should not be split by pre-tokenization.
	const maxSpecialLen = 32 // reasonable max for special tokens like </s>, [INST], etc.

	bestID := -1
	bestLen := 0

	for end := min(len(text), maxSpecialLen); end > 1; end-- {
		candidate := text[:end]
		if id, ok := b.vocab[candidate]; ok {
			// Since we already verified the first character is a special-token prefix,
			// any multi-char vocab match starting with that prefix is a special token.
			if end > bestLen {
				bestID = id
				bestLen = end
			}
		}
	}

	if bestLen > 0 {
		return bestID, bestLen
	}
	return 0, 0
}

// encodeCharacters encodes a string by encoding each character individually.
// Used as fallback when trie/map matching fails and UNKID is not configured.
func (b *BPE) encodeCharacters(text string) []int {
	ids := make([]int, 0, len(text))
	remaining := text
	for len(remaining) > 0 {
		r, width := utf8.DecodeRuneInString(remaining)
		chunk := remaining[:width]
		if id, ok := b.vocab[chunk]; ok {
			ids = append(ids, id)
		} else if b.unkID >= 0 {
			ids = append(ids, int(b.unkID))
		} else if len(chunk) == 1 && chunk[0] < 128 {
			// ASCII char with no vocab entry — encode as single-byte token (id=byte value).
			ids = append(ids, int(chunk[0]))
		} else {
			// Multi-byte char with no vocab entry — encode as negative ID to distinguish from real token IDs.
			// Detokenize checks for negative IDs and converts back via utf8.EncodeRune.
			ids = append(ids, -(int(r) + 1))
		}
		remaining = remaining[width:]
	}
	return ids
}

// encodeFragment encodes a single pre-tokenized fragment via trie + map fallback.
func (b *BPE) encodeFragment(fragment string) []int {
	if b.trie == nil {
		return b.fallback(fragment)
	}

	content := fragment

	id, length := b.trie.matchLongest([]byte(content))

	// Re-match the content if not fully matched.
	if id < 0 || (length > 0 && length != len(content)) {
		id, length = b.trie.matchLongest([]byte(content))
	}

	if id >= 0 && length > 0 && length == len(content) {
		return []int{id}
	}

	dst := make([]int, 0, 8)
	if id >= 0 && length > 0 {
		dst = append(dst, id)
	}
	remaining := content[length:]

	// encodeRemaining handles the main loop: try trie match, then vocab map,
	// then fall back to character-by-character encoding.
	for len(remaining) > 0 {
		id, length := b.trie.matchLongest([]byte(remaining))
		if id >= 0 && length > 0 {
			dst = append(dst, id)
			remaining = remaining[length:]
			continue
		}

		var matched bool

		for end := len(remaining); end > 0; end-- {
			if id, ok := b.vocab[remaining[:end]]; ok {
				dst = append(dst, id)
				remaining = remaining[end:]
				matched = true
				break
			}
		}
		if !matched {
			if b.unkDisabled && len(remaining) > 0 {
				// UNK explicitly disabled — return empty to indicate unknown content.
				dst = nil
				break
			} else if b.unkID >= 0 && len(remaining) > 0 {
				dst = append(dst, int(b.unkID))
				_, width := utf8.DecodeRuneInString(remaining)
				remaining = remaining[width:]
			} else if len(remaining) > 0 {
				// No UNK configured — encode character-by-character via vocab lookup.
				// This ensures we never lose text even without an explicit UNK token.
				r, _ := utf8.DecodeRuneInString(remaining)
				s := string(r)
				if id, ok := b.vocab[s]; ok {
					dst = append(dst, id)
				} else if len(s) == 1 && s[0] < 128 {
					// ASCII char with no vocab entry — encode as single-byte token (id=byte value).
					// Detokenize will reconstruct the character from this byte value.
					dst = append(dst, int(s[0]))
				} else {
					// Multi-byte char with no vocab entry — encode as negative ID to distinguish from real token IDs.
					// Detokenize checks for negative IDs and converts back via utf8.EncodeRune.
					dst = append(dst, -(int(r) + 1))
				}
				remaining = remaining[len(s):]
			}
		}
	}

	if len(dst) == 0 && fragment != "" {
		return b.encodeWhitespace(fragment)
	}
	return dst
}

// fallback encodes via vocabulary map only (no trie). Always returns at least one token per non-empty fragment.
func (b *BPE) fallback(fragment string) []int {
	dst := make([]int, 0, 8)
	remaining := fragment
	for len(remaining) > 0 {
		var matched bool
		for end := len(remaining); end > 0; end-- {
			if id, ok := b.vocab[remaining[:end]]; ok {
				dst = append(dst, id)
				remaining = remaining[end:]
				matched = true
				break
			}
		}
		if !matched {
			if b.unkID >= 0 {
				dst = append(dst, int(b.unkID))
			} else if len(remaining) > 0 {
				// No UNK configured — encode character-by-character via vocab lookup.
				r, _ := utf8.DecodeRuneInString(remaining)
				s := string(r)
				if id, ok := b.vocab[s]; ok {
					dst = append(dst, id)
				} else if len(s) == 1 && s[0] < 128 {
					// ASCII byte fallback: encode as single-byte token (id=byte value).
					dst = append(dst, int(s[0]))
				} else {
					// Multi-byte char with no vocab entry — encode as negative ID to distinguish from real token IDs.
					// Detokenize checks for negative IDs and converts back via utf8.EncodeRune.
					dst = append(dst, -(int(r) + 1))
				}
			}
			_, width := utf8.DecodeRuneInString(remaining)
			remaining = remaining[width:]
		}
	}
	return dst
}

// encodeWhitespace handles whitespace-only fragments by looking up
// literal whitespace tokens in the vocab, falling back to individual character encoding.
func (b *BPE) encodeWhitespace(fragment string) []int {
	for _, w := range []string{" ", "\t", "\n", "\r"} {
		if id, ok := b.vocab[w]; ok {
			count := 0
			for _, c := range fragment {
				if string(c) == w {
					count++
				}
			}
			if count > 0 {
				ids := make([]int, 0, count)
				for i := 0; i < count; i++ {
					ids = append(ids, id)
				}
				return ids
			}
		}
	}

	// No literal whitespace token found — encode each character individually.
	if b.unkDisabled {
		return nil // UNK explicitly disabled — no fallback encoding allowed.
	}

	ids := make([]int, 0, len(fragment))
	remaining := fragment
	for len(remaining) > 0 {
		r, width := utf8.DecodeRuneInString(remaining)
		s := string(r)
		if id, ok := b.vocab[s]; ok {
			ids = append(ids, id)
		} else if b.unkID >= 0 {
			ids = append(ids, int(b.unkID))
		} else if len(s) == 1 && s[0] < 128 {
			// ASCII byte fallback: encode as single-byte token (id=byte value).
			ids = append(ids, int(s[0]))
		} else {
			// Multi-byte char with no vocab entry — encode as negative ID to distinguish from real token IDs.
			// Detokenize checks for negative IDs and converts back via utf8.EncodeRune.
			ids = append(ids, -(int(r) + 1))
		}
		remaining = remaining[width:]
	}
	return ids
}

// Detokenize converts token IDs back to text, matching llama.cpp's BPE detokenization:
//   - Space prefix characters (Ġ → space) mark word boundaries in the vocabulary.
//   - BOS/EOS tokens are stripped from output.
//   - Pure punctuation tokens attach directly to previous content if it ends with a letter/digit.
//   - Adjacent non-ASCII tokens (CJK, emoji) stay concatenated without spaces.

// isLatinRune reports whether r is a Latin-script character (basic multilingual plane,
// primarily A-Z, a-z, and common accented Latin characters used in European languages).
// CJK, Cyrillic, Arabic, and other scripts return false so heuristic spacing doesn't
// insert spaces between adjacent non-Latin tokens.
func isLatinRune(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') ||
		(r >= 0x00C0 && r <= 0x024F) // Latin Extended-A and Latin Extended-B
}

func (b *BPE) Detokenize(ids []int) string {
	if b == nil || len(ids) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(ids) * 4) // heuristic: ~4 chars per token on average

	prevTok := ""
	for _, id := range ids {
		tok, ok := b.invVocab[id]

		// Handle BOS/EOS tokens first (skip them even if not in vocab).
		if (b.addBOS && int(b.bosID) == id) || (b.addEOS && int(b.eosID) == id) {
			continue
		}

		// If token is not in vocabulary, treat it as a raw byte value or rune code point.
		if !ok {
			if id < 0 {
				// Negative IDs represent Unicode code points for characters not in the vocabulary.
				// Decode: rune = -id - 1
				r := rune(-id - 1)
				// When spaceChar is auto-detected but prefixed tokens aren't all in vocab,
				// whitespace chars leak through as raw rune IDs. Convert them back to their
				// literal ASCII form for round-trip fidelity.
				if r == b.spaceChar {
					sb.WriteByte(' ')
				} else if unicode.IsSpace(r) {
					switch r {
					case '\t':
						sb.WriteByte('\t')
					case '\n', '\r':
						sb.WriteByte(byte(r))
					default:
						sb.WriteByte(' ')
					}
				} else {
					var buf [4]byte
					n := utf8.EncodeRune(buf[:], r)
					sb.Write(buf[:n])
				}
			} else if id >= 0 && id < 256 {
				sb.WriteByte(byte(id))
			}
			// Unknown positive IDs are skipped silently to avoid panics.
			prevTok = ""
			continue
		}

		// Handle tokens that start with the space prefix character.
		// In BPE models, a token like "Ġworld" means " world" — the Ġ marks a word boundary.
		if b.spaceChar != 0 && len(tok) > 0 {
			r, w := utf8.DecodeRuneInString(tok)
			if int(r) == int(b.spaceChar) {
				content := tok[w:]
				// If the token is just the space prefix (no content after it), output a literal space.
				if len(content) == 0 {
					sb.WriteByte(' ')
				} else {
					// Add leading space only if there's previous content to separate from.
					if sb.Len() > 0 {
						sb.WriteByte(' ')
					}
					sb.WriteString(content)
				}
				prevTok = content
				continue
			}
		}

		// For non-space-prefixed tokens, add spaces based on token boundaries.
		// This heuristic ONLY applies when no spaceChar is configured (simple test vocabularies).
		// Production BPE models with spaceChar handle spacing via the pre-tokenizer's
		// space prefix mechanism — adding heuristic spaces there would double-space.
		if b.spaceChar == 0 && prevTok != "" {
			lastR, lastW := utf8.DecodeLastRuneInString(prevTok)
			firstR, firstW := utf8.DecodeRuneInString(tok)

			// Check if previous token ends with a Latin letter or digit.
			prevEndsWord := lastW > 0 &&
				(unicode.IsLetter(lastR) || unicode.IsDigit(lastR)) && isLatinRune(lastR)

			// Check if current token starts with a Latin letter or digit.
			currStartsWord := firstW > 0 &&
				(unicode.IsLetter(firstR) || unicode.IsDigit(firstR)) && isLatinRune(firstR)

			if prevEndsWord && currStartsWord {
				sb.WriteByte(' ')
			} else if !prevEndsWord && currStartsWord {
				// Previous token ends with punctuation or non-word char, current starts a word.
				// Add space to separate (e.g., "," + "world" → ", world").
				lastIsPunct := lastW > 0 && unicode.IsPunct(lastR)
				if lastIsPunct {
					sb.WriteByte(' ')
				}
			} else if prevEndsWord && !currStartsWord {
				// Previous token ends with word char, current starts with punctuation.
				// Attach without space (e.g., "hello" + "," → "hello,").
			}
		}

		sb.WriteString(tok)
		prevTok = tok
	}

	return sb.String()
}

// Count returns the number of tokens in text, including BOS/EOS if configured.
// Uses zero slice allocation for performance — equivalent to len(EncodeIDs(text)).
func (b *BPE) Count(text string) int {
	if b == nil || len(text) == 0 {
		return 0
	}

	count := 0
	if b.addBOS && b.bosID >= 0 {
		count++
	}

	if _, ok := b.vocab[text]; ok {
		count++
	} else {
		splits := preTokenize(text, b.preType, b.spaceChar)
		for _, s := range splits {
			count += countFragment(b, s.Text)
		}
	}

	if b.addEOS && b.eosID >= 0 {
		count++
	}

	return count
}

// TokenCount is an alias for Count, provided for compatibility with other tokenizer interfaces.
func (b *BPE) TokenCount(text string) int {
	return b.Count(text)
}

// Tokens returns the token strings corresponding to each ID in EncodeIDs(text).
// Equivalent to mapping each ID through invVocab. Returns nil for empty input or nil receiver.
func (b *BPE) Tokens(text string) []string {
	if b == nil {
		return nil
	}
	ids := b.EncodeIDs(text)
	if len(ids) == 0 {
		return nil
	}
	tokens := make([]string, 0, len(ids))
	for _, id := range ids {
		tokens = append(tokens, b.invVocab[id])
	}
	return tokens
}

// HasToken reports whether the given token string exists in the vocabulary.
func (b *BPE) HasToken(tok string) bool {
	if b == nil {
		return false
	}
	_, ok := b.vocab[tok]
	return ok
}

// TokenID returns the integer ID for a token string, or -1 if not found in vocabulary.
func (b *BPE) TokenID(tok string) int {
	if b == nil {
		return -1
	}
	id, ok := b.vocab[tok]
	if !ok {
		return -1
	}
	return id
}

// Type returns "bpe" to identify this tokenizer type.
func (b *BPE) Type() string { return "bpe" }

// BOSID returns the beginning-of-sequence token ID, or -1 if not configured.
func (b *BPE) BOSID() int64 { return b.bosID }

// EOSID returns the end-of-sentence token ID, or -1 if not configured.
func (b *BPE) EOSID() int64 { return b.eosID }

// SetCache enables caching of encoded results with the given capacity (number of entries).
// Pass 0 or negative to disable caching. Cached results are invalidated by ClearCache().
func (b *BPE) SetCache(size int) {
	if b == nil || size <= 0 {
		return
	}
	if b.cache == nil {
		b.cache = make(map[string][]int, size)
	}
}

// ClearCache removes all cached encoding results, freeing memory. Use after significant
// vocabulary changes or when caching is no longer needed.
func (b *BPE) ClearCache() {
	if b == nil || b.cache == nil {
		return
	}
	for k := range b.cache {
		delete(b.cache, k)
	}
}

// countFragment counts tokens in a single fragment without allocating.
func countFragment(b *BPE, fragment string) int {
	count := 0
	if b.trie == nil {
		return countFallback(b, fragment)
	}

	remaining := fragment
	for len(remaining) > 0 {
		id, length := b.trie.matchLongest([]byte(remaining))
		if id >= 0 && length > 0 {
			count++
			remaining = remaining[length:]
			continue
		}
		var matched bool
		for end := len(remaining); end > 0; end-- {
			if _, ok := b.vocab[remaining[:end]]; ok {
				count++
				remaining = remaining[end:]
				matched = true
				break
			}
		}
		if !matched {
			if b.unkDisabled && len(remaining) > 0 {
				// UNK explicitly disabled — return 0 to indicate unknown content.
				count = 0
				return count
			} else if b.unkID >= 0 {
				count++
			} else if len(remaining) > 0 {
				// No UNK configured — encode character-by-character via vocab lookup.
				r, _ := utf8.DecodeRuneInString(remaining)
				s := string(r)
				if _, ok := b.vocab[s]; ok {
					count++
				} else if len(s) == 1 && s[0] < 128 {
					// ASCII byte fallback: encode as single-byte token (id=byte value).
					count++
				} else {
					// Multi-byte char with no vocab entry — count each byte.
					for i := 0; i < len(s); i++ {
						count++
					}
				}
			}
			_, width := utf8.DecodeRuneInString(remaining)
			remaining = remaining[width:]
		}
	}
	if count == 0 {
		return countWhitespace(b, fragment)
	}
	return count
}

func countFallback(b *BPE, fragment string) int {
	count := 0
	remaining := fragment
	for len(remaining) > 0 {
		var matched bool
		for end := len(remaining); end > 0; end-- {
			if _, ok := b.vocab[remaining[:end]]; ok {
				count++
				remaining = remaining[end:]
				matched = true
				break
			}
		}
		if !matched {
			if b.unkDisabled && len(remaining) > 0 {
				// UNK explicitly disabled — return 0 to indicate unknown content.
				count = 0
				return count
			} else if b.unkID >= 0 {
				count++
			} else if len(remaining) > 0 {
				// No UNK configured — encode character-by-character via vocab lookup.
				r, _ := utf8.DecodeRuneInString(remaining)
				s := string(r)
				if _, ok := b.vocab[s]; ok {
					count++
				} else if len(s) == 1 && s[0] < 128 {
					// ASCII byte fallback: encode as single-byte token (id=byte value).
					count++
				} else {
					// Multi-byte char with no vocab entry — count each byte.
					for i := 0; i < len(s); i++ {
						count++
					}
				}
			}
			_, width := utf8.DecodeRuneInString(remaining)
			remaining = remaining[width:]
		}
	}
	if count == 0 {
		return countWhitespace(b, fragment)
	}
	return count
}

func countWhitespace(b *BPE, fragment string) int {
	var count int
	for _, w := range []string{" ", "\t", "\n", "\r"} {
		if _, ok := b.vocab[w]; ok {
			for _, c := range fragment {
				if string(c) == w {
					count++
				}
			}
		}
	}
	return count
}
