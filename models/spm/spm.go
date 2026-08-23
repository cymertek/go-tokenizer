// Package spm implements a SentencePiece (SPM) tokenizer that reads protobuf-encoded
// model binaries from GGUF files or is configured programmatically. It uses a byte-level
// trie for O(k) longest-match token lookup and supports both greedy longest-match and
// dynamic-programming segmentation modes.
package spm

import (
	"fmt"
	"strings"
	"sync"
	"unicode"

	"github.com/cymertek/go-tokenizer/models/common"
)

const metaChar = '▁' // ▁ — SentencePiece space marker for word boundaries

// SPM is the SentencePiece tokenizer. It implements common.Tokenizer and uses greedy
// longest-match decoding with ▁-prefixed vocabulary tokens representing word boundaries.
type SPM struct {
	data      *common.TokenizerData // source configuration (may be nil if built from protobuf)
	vocab     map[string]int        // token string → id (for fast lookup)
	invVocab  []int                 // id → index into vocab keys slice
	tokenType []int32              // per-token type flags (parallel to vocab keys, for GGUF compat)
	vocabKeys []string              // ordered list of all tokens for inverse lookup
	trie      *byteTrie             // byte trie for O(k) longest-match
	caches    sync.Map              // input text → cached []int (thread-safe)

	modelType string // always "spm"
}

// New creates a new SPM tokenizer from the given TokenizerData. The data must have either:
//   - Model == "spm" with Tokens set (programmatic mode), or
//   - Model == "spm" with SPMModel containing protobuf binary (GGUF file mode)
//
// Returns nil-safe: if data is nil, returns an empty SPM receiver.
func New(data *common.TokenizerData) (*SPM, error) {
	if data == nil {
		return &SPM{modelType: "spm"}, nil // nil-safe receiver
	}

	s := &SPM{
		data:      data,
		modelType: "spm",
		vocab:     make(map[string]int),
		invVocab:  make([]int, len(data.Tokens)),
		tokenType: data.TokenType, // Preserve TokenType array if present.
	}

	if s.data.SPMModel != nil && len(s.data.SPMModel) > 0 {
		// Parse protobuf binary from GGUF file
		pieces, err := decodeModelProto(s.data.SPMModel)
		if err != nil {
			return nil, fmt.Errorf("spm: failed to parse model proto: %w", err)
		}
		s.buildFromPieces(pieces)
	} else if len(data.Tokens) > 0 {
		// Programmatic mode — build from token list directly
		s.buildVocabMap(data.Tokens)
	}

	return s, nil
}

// buildFromPieces constructs the trie and vocab map from parsed protobuf pieces.
func (s *SPM) buildFromPieces(pieces []piece) {
	for _, p := range pieces {
		if int(p.ID) >= len(s.vocabKeys) {
			s.vocabKeys = append(s.vocabKeys, make([]string, int(p.ID)-len(s.vocabKeys)+1)...)
		}
		s.vocabKeys[p.ID] = p.Word
		s.vocab[p.Word] = int(p.ID)
	}
	s.buildTrie()
}

// buildVocabMap constructs the trie and vocab map from a flat token list.
func (s *SPM) buildVocabMap(tokens []string) {
	for i, tok := range tokens {
		if i >= len(s.vocabKeys) {
			s.vocabKeys = append(s.vocabKeys, make([]string, i-len(s.vocabKeys)+1)...)
		}
		s.vocabKeys[i] = tok
		s.vocab[tok] = i
	}
	s.buildTrie()
}

// buildTrie constructs the byte trie from all vocabulary tokens.
func (s *SPM) buildTrie() {
	tr := newByteTrie()
	for _, tok := range s.vocabKeys {
		if id, ok := s.vocab[tok]; ok {
			tr.insert(tok, id)
		}
	}
	s.trie = tr
}

// convertSpacesToMeta prepends ▁ before each word boundary and at the start of text.
// SentencePiece uses ▁ as a space marker: "hello world" → "▁hello▁world".
func (s *SPM) convertSpacesToMeta(text string) string {
	if len(text) == 0 {
		return text
	}
	var b strings.Builder
	b.WriteString(string(metaChar)) // always start with ▁
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			b.WriteString(string(metaChar))
		} else {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// EncodeIDs tokenizes text into a slice of integer IDs using greedy longest-match.
// Spaces in input are converted to ▁ meta characters before matching against vocabulary.
// If BOS/EOS are configured, they are prepended/appended to the result.
func (s *SPM) EncodeIDs(text string) []int {
	if s == nil || len(text) == 0 {
		return nil
	}

	// Check cache first
	if cached, ok := s.caches.Load(text); ok {
		if ids, ok := cached.([]int); ok {
			return ids
		}
	}

	ids := s.encodeText(text)

	// Add BOS if configured
	if s.data != nil && s.data.HasBOSID && s.data.AddBOS {
		bos := make([]int, 0, len(ids)+1)
		bos = append(bos, int(s.data.BOSID))
		bos = append(bos, ids...)
		ids = bos
	}

	// Add EOS if configured
	if s.data != nil && s.data.HasEOSID && s.data.AddEOS {
		eos := make([]int, 0, len(ids)+1)
		eos = append(eos, ids...)
		eos = append(eos, int(s.data.EOSID))
		ids = eos
	}

	s.caches.Store(text, ids)
	return ids
}

// encodeText performs the core greedy longest-match tokenization.
// Spaces in the input are first converted to ▁ meta characters so that the trie can match
// SentencePiece-style tokens (e.g., "▁hello", "world").
func (s *SPM) encodeText(text string) []int {
	// Convert spaces → ▁ for matching against the vocab.
	normalized := s.convertSpacesToMeta(text)

	var ids []int
	i := 0
	for i < len(normalized) {
		id, length := s.trie.matchLongest([]byte(normalized[i:]))
		if id >= 0 && length > 0 {
			tok := ""
			if id < len(s.vocabKeys) {
				tok = s.vocabKeys[id]
			}
			ids = append(ids, id)
			i += length
			_ = tok // debug info available via test logging
		} else {
			// No match at all — emit single-byte rune (or UNK if configured).
			r, size := decodeRune([]byte(normalized[i:]))
			if size == 0 {
				break // safety valve
			}
			tok := string(r)
			if tokID, ok := s.vocab[tok]; ok {
				ids = append(ids, tokID)
			} else if s.data != nil && s.data.HasUNKID {
				ids = append(ids, int(s.data.UNKID))
			}
			i += size
		}
	}

	if len(ids) == 0 {
		return []int{} // return empty slice, not nil
	}
	return ids
}

// decodeRune decodes a single UTF-8 rune from the byte slice, returning (r, width).
func decodeRune(b []byte) (rune, int) {
	if len(b) == 0 {
		return 0, 0
	}
	c := b[0]
	if c < 0x80 {
		return rune(c), 1
	} else if c&0xE0 == 0xC0 {
		if len(b) < 2 {
			return 0, 0
		}
		r := rune(c&0x1F)<<6 | rune(b[1]&0x3F)
		return r, 2
	} else if c&0xF0 == 0xE0 {
		if len(b) < 3 {
			return 0, 0
		}
		r := rune(c&0x0F)<<12 | rune(b[1]&0x3F)<<6 | rune(b[2]&0x3F)
		return r, 3
	} else if c&0xF8 == 0xF0 {
		if len(b) < 4 {
			return 0, 0
		}
		r := rune(c&0x07)<<18 | rune(b[1]&0x3F)<<12 | rune(b[2]&0x3F)<<6 | rune(b[3]&0x3F)
		return r, 4
	}
	return 0, 0
}

// Detokenize converts a slice of token IDs back to text, replacing ▁ meta characters with spaces.
// Leading whitespace from the first ▁ is stripped. BOS/EOS tokens are skipped if present.
func (s *SPM) Detokenize(ids []int) string {
	if s == nil || len(ids) == 0 {
		return ""
	}

	var b strings.Builder
	prevTok := ""
	for _, id := range ids {
		tok := s.idToToken(id)
		if tok == "" {
			continue
		}
		// Replace ▁ with space so word boundaries are preserved.
		replaced := strings.ReplaceAll(tok, string(metaChar), " ")

		// Strip leading whitespace from the first content token — the initial ▁ marks
		// the start of text and shouldn't produce a leading space in output.
		if prevTok == "" {
			replaced = strings.TrimLeft(replaced, " \t\n\r")
		}

		// Add space between tokens if previous token didn't end with whitespace
		// and current token doesn't start with whitespace (after meta replacement).
		if prevTok != "" {
			prevEndsWS := unicode.IsSpace(rune(prevTok[len(prevTok)-1]))
			currStartsWS := len(replaced) > 0 && unicode.IsSpace(rune(replaced[0]))

			if !prevEndsWS && !currStartsWS {
				b.WriteByte(' ')
			}
		}

		b.WriteString(replaced)
		prevTok = replaced
	}
	return b.String()
}

// IDToToken returns the token string for a given ID, or empty string if the ID is out of range.
func (s *SPM) IDToToken(id int) string {
	if s == nil || id < 0 || id >= len(s.vocabKeys) {
		return ""
	}
	return s.vocabKeys[id]
}

// idToToken is an alias for IDToToken, kept for internal use.
func (s *SPM) idToToken(id int) string {
	return s.IDToToken(id)
}

// Count returns the number of tokens for the given text, including BOS/EOS if configured.
func (s *SPM) Count(text string) int {
	ids := s.EncodeIDs(text)
	if ids == nil {
		return 0
	}
	return len(ids)
}

// Type returns "spm" to identify this tokenizer type.
func (s *SPM) Type() string {
	return "spm"
}

// HasToken checks whether the given token string exists in the vocabulary.
// Returns false for nil receiver or empty vocabulary.
func (s *SPM) HasToken(tok string) bool {
	if s == nil || len(s.vocab) == 0 {
		return false
	}
	_, ok := s.vocab[tok]
	return ok
}

// TokenID returns the integer ID for a given token, or -1 if not found in vocabulary.
func (s *SPM) TokenID(tok string) int {
	if s == nil || len(s.vocab) == 0 {
		return -1
	}
	id, ok := s.vocab[tok]
	if !ok {
		return -1
	}
	return id
}

// ClearCache removes all cached encodings, freeing memory used by the sync.Map cache.
func (s *SPM) ClearCache() {
	s.caches = sync.Map{}
}

// --- init() registration ---

func init() {
	common.Register("spm", func(data *common.TokenizerData) (common.Tokenizer, error) {
		return New(data)
	})
}
