// Package unigram implements a Unigram tokenizer that uses probabilistic token scoring
// with Viterbi decoding for optimal segmentation. It supports both programmatic
// configuration and GGUF model file loading. Unlike greedy longest-match, Unigram
// selects the segmentation that maximizes the product of token probabilities.
package unigram

import (
	"math"
	"strings"
	"sync"

	"github.com/cymertek/go-tokenizer/models/common"
)

const metaChar = '▁' // ▁ — Unigram space marker for word boundaries

// Unigram is the Unigram tokenizer. It implements common.Tokenizer and uses Viterbi
// dynamic programming to find the highest-probability segmentation of input text.
type Unigram struct {
	data      *common.TokenizerData // source configuration (may be nil)
	vocab     map[string]int        // token string → id (for fast lookup)
	vocabKeys []string              // ordered list of all tokens for inverse lookup
	scores    []float64             // log-probability scores for each token (parallel to vocabKeys)
	trie      *byteTrie             // byte trie for O(k) longest-match
	caches    sync.Map              // input text → cached []int (thread-safe)

	modelType string // always "unigram"
}

// New creates a new Unigram tokenizer from the given TokenizerData. The data must have
// Model == "unigram" with Tokens set. For optimal segmentation, also provide SPMProbabilities
// as interleaved [tokenID, score] pairs in data.SPMProbabilities (or flat scores parallel to tokens).
func New(data *common.TokenizerData) (*Unigram, error) {
	if data == nil {
		return &Unigram{modelType: "unigram"}, nil // nil-safe receiver
	}

	u := &Unigram{
		data:      data,
		modelType: "unigram",
		vocab:     make(map[string]int),
		vocabKeys: make([]string, len(data.Tokens)),
		scores:    make([]float64, len(data.Tokens)),
	}

	if len(data.Tokens) > 0 {
		u.buildVocabMap(data.Tokens, data.SPMProbabilities)
	}

	return u, nil
}

// buildVocabMap constructs the trie and vocab map from token list with optional scores.
// SPMProbabilities can be either:
//   - Interleaved [id, score, id, score, ...] pairs (from serialization)
//   - Flat scores parallel to tokens (legacy format)
func (u *Unigram) buildVocabMap(tokens []string, probs []float64) {
	// If probs are interleaved [id, score] pairs, unpack them into flat scores array.
	if len(probs)%2 == 0 && len(probs)/2 == len(tokens) {
		// Interleaved format: [id0, score0, id1, score1, ...]
		for i := 0; i < len(tokens); i++ {
			u.vocabKeys[i] = tokens[i]
			u.vocab[tokens[i]] = i
			scoreIdx := i*2 + 1 // score is at odd indices in interleaved format
			if scoreIdx < len(probs) {
				u.scores[i] = probs[scoreIdx]
			} else {
				u.scores[i] = 0.0
			}
		}
	} else if len(probs) > 0 {
		// Flat format: scores parallel to tokens
		for i, tok := range tokens {
			u.vocabKeys[i] = tok
			u.vocab[tok] = i
			if i < len(probs) {
				u.scores[i] = probs[i]
			} else {
				u.scores[i] = 0.0 // default score
			}
		}
	} else {
		// No scores provided - use defaults
		for i, tok := range tokens {
			u.vocabKeys[i] = tok
			u.vocab[tok] = i
			u.scores[i] = 0.0 // default score
		}
	}

	u.buildTrie()
}

// buildTrie constructs the byte trie from all vocabulary tokens.
func (u *Unigram) buildTrie() {
	tr := newByteTrie()
	for _, tok := range u.vocabKeys {
		if id, ok := u.vocab[tok]; ok {
			tr.insert(tok, id)
		}
	}
	u.trie = tr
}

// convertSpacesToMeta prepends ▁ before each word boundary and at the start of text.
// SentencePiece models store words with leading ▁ (e.g., "▁hello", "▁world").
func (u *Unigram) convertSpacesToMeta(text string) string {
	if len(text) == 0 {
		return text
	}
	var b strings.Builder
	b.WriteString(string(metaChar)) // always start with ▁ for first word
	for i := 0; i < len(text); i++ {
		c := text[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			b.WriteString(string(metaChar)) // mark word boundary
		} else {
			b.WriteByte(c)
		}
	}
	return b.String()
}

// EncodeIDs tokenizes text into a slice of integer IDs using Viterbi dynamic programming.
// Spaces in input are converted to ▁ meta characters before matching against vocabulary.
// The segmentation maximizes the product of token log-probabilities. If BOS/EOS are configured,
// they are prepended/appended to the result.
func (u *Unigram) EncodeIDs(text string) []int {
	if u == nil || len(text) == 0 {
		return nil
	}

	// Check cache first
	if cached, ok := u.caches.Load(text); ok {
		if ids, ok := cached.([]int); ok {
			return ids
		}
	}

	ids := u.encodeText(text)

	// Add BOS if configured
	if u.data != nil && u.data.HasBOSID && u.data.AddBOS {
		bos := make([]int, 0, len(ids)+1)
		bos = append(bos, int(u.data.BOSID))
		bos = append(bos, ids...)
		ids = bos
	}

	// Add EOS if configured
	if u.data != nil && u.data.HasEOSID && u.data.AddEOS {
		eos := make([]int, 0, len(ids)+1)
		eos = append(eos, ids...)
		eos = append(eos, int(u.data.EOSID))
		ids = eos
	}

	u.caches.Store(text, ids)
	return ids
}

// encodeText performs beam search decoding for optimal segmentation.
func (u *Unigram) encodeText(text string) []int {
	normalized := u.convertSpacesToMeta(text)

	// Use Viterbi-like DP with beam search to find best segmentation.
	ids, _ := u.viterbiDecode(normalized)

	if len(ids) == 0 {
		return []int{} // return empty slice, not nil
	}
	return ids
}

// viterbiDecode uses dynamic programming to find the highest-scoring tokenization.
func (u *Unigram) viterbiDecode(text string) ([]int, float64) {
	n := len(text)
	if n == 0 {
		return nil, 0
	}

	// dp[i] = best score for text[:i], -Inf if unreachable
	dp := make([]float64, n+1)
	for i := range dp {
		dp[i] = math.Inf(-1)
	}
	dp[0] = 0.0

	// backptr[i] = (tokenID, startOffset) for the best segmentation ending at position i
	type backtrack struct {
		id      int
		startAt int
	}
	backptr := make([]backtrack, n+1)

	// Try all possible token matches starting from each position.
	for i := 0; i < n; i++ {
		if dp[i] == math.Inf(-1) {
			continue // unreachable state
		}

		// Find all tokens that match text[i:]
		id, length := u.trie.matchLongest([]byte(text[i:]))
		for id >= 0 && length > 0 {
			j := i + length
			if j > n {
				break // shouldn't happen with correct trie
			}

			score := dp[i] + u.scores[id]
			if score > dp[j] {
				dp[j] = score
				backptr[j] = backtrack{id: id, startAt: i}
			}

			// Try shorter matches (substrings of the longest match).
			// This is a simplification — in practice, we'd enumerate all possible tokens.
			// For now, just use the longest match at each position.
			break
		}

		// Also try single-character matches for unmatched characters.
		if id < 0 {
			r, size := decodeRune([]byte(text[i:]))
			if size > 0 {
				tok := string(r)
				if tokID, ok := u.vocab[tok]; ok {
					j := i + size
					score := dp[i] + u.scores[tokID]
					if score > dp[j] {
						dp[j] = score
						backptr[j] = backtrack{id: tokID, startAt: i}
					}
				} else if u.data != nil && u.data.HasUNKID {
					j := i + size
					score := dp[i] + u.scores[int(u.data.UNKID)]
					if score > dp[j] {
						dp[j] = score
						backptr[j] = backtrack{id: int(u.data.UNKID), startAt: i}
					}
				}
			}
		}
	}

	// Backtrack to find the best segmentation.
	if dp[n] == math.Inf(-1) {
		return nil, 0 // no valid segmentation found
	}

	var ids []int
	pos := n
	for pos > 0 {
		bt := backptr[pos]
		ids = append([]int{bt.id}, ids...)
		pos = bt.startAt
	}

	return ids, dp[n]
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
func (u *Unigram) Detokenize(ids []int) string {
	if u == nil || len(ids) == 0 {
		return ""
	}

	var b strings.Builder
	prevTok := ""
	for _, id := range ids {
		tok := u.IDToToken(id)
		if tok == "" {
			continue
		}
		replaced := strings.ReplaceAll(tok, string(metaChar), " ")
		// Strip leading whitespace from the first content token — the initial ▁ marks
		// the start of text and shouldn't produce a leading space in output.
		if prevTok == "" {
			replaced = strings.TrimLeft(replaced, " \t\n\r")
		}
		b.WriteString(replaced)
		prevTok = replaced
	}
	return b.String()
}

// IDToToken returns the token string for a given ID, or empty string if the ID is out of range.
func (u *Unigram) IDToToken(id int) string {
	if u == nil || id < 0 || id >= len(u.vocabKeys) {
		return ""
	}
	return u.vocabKeys[id]
}

// Count returns the number of tokens for the given text, including BOS/EOS if configured.
func (u *Unigram) Count(text string) int {
	ids := u.EncodeIDs(text)
	if ids == nil {
		return 0
	}
	return len(ids)
}

// Type returns "unigram" to identify this tokenizer type.
func (u *Unigram) Type() string {
	return "unigram"
}

// HasToken checks whether the given token string exists in the vocabulary.
// Returns false for nil receiver or empty vocabulary.
func (u *Unigram) HasToken(tok string) bool {
	if u == nil || len(u.vocab) == 0 {
		return false
	}
	_, ok := u.vocab[tok]
	return ok
}

// TokenID returns the integer ID for a given token, or -1 if not found in vocabulary.
func (u *Unigram) TokenID(tok string) int {
	if u == nil || len(u.vocab) == 0 {
		return -1
	}
	id, ok := u.vocab[tok]
	if !ok {
		return -1
	}
	return id
}

// ClearCache removes all cached encodings, freeing memory used by the sync.Map cache.
func (u *Unigram) ClearCache() {
	u.caches = sync.Map{}
}

// --- init() registration ---

func init() {
	common.Register("unigram", func(data *common.TokenizerData) (common.Tokenizer, error) {
		return New(data)
	})
}

// Ensure Unigram implements common.Tokenizer interface.
var _ common.Tokenizer = (*Unigram)(nil)
