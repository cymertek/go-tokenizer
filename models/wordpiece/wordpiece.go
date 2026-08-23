// Package wordpiece implements a WordPiece tokenizer used by BERT and similar models.
// It uses ## prefix for continuation subwords (e.g., "playing" → ["play", "##ing"])
// and performs greedy longest-match decoding with whitespace pre-tokenization.
package wordpiece

import (
	"strings"
	"sync"

	"github.com/cymertek/go-tokenizer/models/common"
)

const continuationPrefix = "##" // WordPiece continuation marker for subword tokens

// WordPiece is the WordPiece tokenizer. It implements common.Tokenizer and uses ## prefix
// for continuation subwords (e.g., "playing" → ["play", "##ing"]). Text is pre-tokenized
// on whitespace, then each fragment is greedily matched against the vocabulary.
type WordPiece struct {
	data      *common.TokenizerData // source configuration (may be nil)
	vocab     map[string]int        // token string → id (for fast lookup)
	vocabKeys []string              // ordered list of all tokens for inverse lookup
	trie      *byteTrie             // byte trie for O(k) longest-match
	caches    sync.Map              // input text → cached []int (thread-safe)

	modelType string // always "wordpiece"
}

// New creates a new WordPiece tokenizer from the given TokenizerData. The data must have
// Model == "wordpiece" with Tokens set. Vocabulary should include special tokens like
// [CLS], [SEP], [UNK], and ##-prefixed subwords for proper segmentation.
func New(data *common.TokenizerData) (*WordPiece, error) {
	if data == nil {
		return &WordPiece{modelType: "wordpiece"}, nil // nil-safe receiver
	}

	w := &WordPiece{
		data:      data,
		modelType: "wordpiece",
		vocab:     make(map[string]int),
		vocabKeys: make([]string, len(data.Tokens)),
	}

	if len(data.Tokens) > 0 {
		w.buildVocabMap(data.Tokens)
	}

	return w, nil
}

// buildVocabMap constructs the trie and vocab map from a flat token list.
func (w *WordPiece) buildVocabMap(tokens []string) {
	for i, tok := range tokens {
		w.vocabKeys[i] = tok
		w.vocab[tok] = i
	}
	w.buildTrie()
}

// buildTrie constructs the byte trie from all vocabulary tokens.
func (w *WordPiece) buildTrie() {
	tr := newByteTrie()
	for _, tok := range w.vocabKeys {
		if id, ok := w.vocab[tok]; ok {
			tr.insert(tok, id)
		}
	}
	w.trie = tr
}

// EncodeIDs tokenizes text into a slice of integer IDs using greedy longest-match with
// whitespace pre-tokenization. Each word fragment is matched against the vocabulary, with
// ##-prefixed tokens treated as continuation subwords. If BOS/EOS are configured, they are
// prepended/appended to the result.
func (w *WordPiece) EncodeIDs(text string) []int {
	if w == nil || len(text) == 0 {
		return nil
	}

	// Check cache first
	if cached, ok := w.caches.Load(text); ok {
		if ids, ok := cached.([]int); ok {
			return ids
		}
	}

	ids := w.encodeText(text)

	// Add BOS if configured
	if w.data != nil && w.data.HasBOSID && w.data.AddBOS {
		bos := make([]int, 0, len(ids)+1)
		bos = append(bos, int(w.data.BOSID))
		bos = append(bos, ids...)
		ids = bos
	}

	// Add EOS if configured
	if w.data != nil && w.data.HasEOSID && w.data.AddEOS {
		eos := make([]int, 0, len(ids)+1)
		eos = append(eos, ids...)
		eos = append(eos, int(w.data.EOSID))
		ids = eos
	}

	w.caches.Store(text, ids)
	return ids
}

// encodeText performs whitespace pre-tokenization followed by greedy longest-match.
func (w *WordPiece) encodeText(text string) []int {
	// Split text into word fragments on whitespace.
	fragments := strings.Fields(text)
	if len(fragments) == 0 {
		return []int{}
	}

	var ids []int
	for fi, frag := range fragments {
		fragIDs := w.tokenizeFragment(frag)
		if fi > 0 && len(fragIDs) > 0 {
			// Insert a space token if the model has one.
			// WordPiece models typically have " " or " ##" as a space token.
			if spaceID, ok := w.vocab[" "]; ok {
				ids = append(ids, spaceID)
			} else if spaceToken, ok := w.findSpaceToken(); ok {
				ids = append(ids, spaceToken)
			}
		}
		ids = append(ids, fragIDs...)
	}

	if len(ids) == 0 {
		return []int{}
	}
	return ids
}

// tokenizeFragment tokenizes a single word fragment using greedy longest-match.
// WordPiece uses ## prefix for continuation subwords (e.g., "playing" → ["play", "##ing"]).
func (w *WordPiece) tokenizeFragment(frag string) []int {
	var ids []int
	i := 0
	for i < len(frag) {
		id, length := w.trie.matchLongest([]byte(frag[i:]))
		if id >= 0 && length > 0 {
			tok := w.vocabKeys[id]
			isCont := strings.HasPrefix(tok, continuationPrefix)

			if isCont {
				// Continuation token — append directly to previous.
				ids = append(ids, id)
				i += length
			} else if len(ids) == 0 || !strings.HasPrefix(w.vocabKeys[ids[len(ids)-1]], continuationPrefix) {
				// Root token (first or after a non-continuation).
				ids = append(ids, id)
				i += length
			} else {
				// Already have a root word, and this is another root — stop here.
				break
			}
		} else {
			// No match at all — emit UNK for remaining characters.
			if w.data != nil && w.data.HasUNKID {
				for j := i; j < len(frag); j++ {
					ids = append(ids, int(w.data.UNKID))
				}
			}
			break
		}
	}

	// After matching root words, check for ## continuation tokens.
	if len(ids) > 0 && !strings.HasPrefix(w.vocabKeys[ids[len(ids)-1]], continuationPrefix) {
		remaining := frag[i:]
		for remaining != "" {
			contID, contLen := w.trie.matchLongest([]byte(continuationPrefix + remaining))
			if contID >= 0 && contLen == len(continuationPrefix)+len(remaining) {
				ids = append(ids, contID)
				remaining = ""
			} else if contID >= 0 && contLen > len(continuationPrefix) {
				// Partial continuation match — take what we can.
				tok := w.vocabKeys[contID]
				ids = append(ids, contID)
				remaining = remaining[len(tok)-len(continuationPrefix):]
			} else {
				break // no more continuations found
			}
		}
	}

	return ids
}

// findSpaceToken looks for a space-related token in the vocabulary.
func (w *WordPiece) findSpaceToken() (int, bool) {
	// Try common space token patterns.
	for _, pattern := range []string{" ", " ##", "##"} {
		if id, ok := w.vocab[pattern]; ok {
			return id, true
		}
	}
	return -1, false
}

// Detokenize converts a slice of token IDs back to text, handling:
//   - ## continuation prefix stripping (e.g., "##ing" → "ing")
//   - Space insertion between root tokens and after continuations
//   - Special token skipping ([CLS], [SEP], <s>, </s>, [PAD], [UNK])
func (w *WordPiece) Detokenize(ids []int) string {
	if w == nil || len(ids) == 0 {
		return ""
	}

	var b strings.Builder
	prevIsContinuation := false
	for _, id := range ids {
		tok := w.IDToToken(id)
		if tok == "" {
			continue
		}

		// Skip special tokens (BOS/EOS markers like [CLS], [SEP], <s>, </s>).
		if isSpecialToken(tok) {
			prevIsContinuation = false
			continue
		}

		isCont := strings.HasPrefix(tok, continuationPrefix)

		if isCont {
			// Continuation token: append directly without space.
			b.WriteString(tok[len(continuationPrefix):])
			prevIsContinuation = true
		} else if prevIsContinuation && b.Len() > 0 {
			// Previous was continuation, this is new word — add space before.
			b.WriteByte(' ')
			b.WriteString(tok)
			prevIsContinuation = false
		} else {
			// New word: add space separator (except for first token).
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(tok)
			prevIsContinuation = false
		}
	}

	return b.String()
}

// isSpecialToken checks if a token is a special marker (BOS/EOS).
func isSpecialToken(tok string) bool {
	switch tok {
	case "[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "[UNK]":
		return true
	default:
		return false
	}
}

// IDToToken returns the token string for a given ID, or empty string if the ID is out of range.
func (w *WordPiece) IDToToken(id int) string {
	if w == nil || id < 0 || id >= len(w.vocabKeys) {
		return ""
	}
	return w.vocabKeys[id]
}

// Count returns the number of tokens for the given text, including BOS/EOS if configured.
func (w *WordPiece) Count(text string) int {
	ids := w.EncodeIDs(text)
	if ids == nil {
		return 0
	}
	return len(ids)
}

// Type returns "wordpiece" to identify this tokenizer type.
func (w *WordPiece) Type() string {
	return "wordpiece"
}

// HasToken checks whether the given token string exists in the vocabulary.
// Returns false for nil receiver or empty vocabulary.
func (w *WordPiece) HasToken(tok string) bool {
	if w == nil || len(w.vocab) == 0 {
		return false
	}
	_, ok := w.vocab[tok]
	return ok
}

// TokenID returns the integer ID for a given token, or -1 if not found in vocabulary.
func (w *WordPiece) TokenID(tok string) int {
	if w == nil || len(w.vocab) == 0 {
		return -1
	}
	id, ok := w.vocab[tok]
	if !ok {
		return -1
	}
	return id
}

// ClearCache removes all cached encodings, freeing memory used by the sync.Map cache.
func (w *WordPiece) ClearCache() {
	w.caches = sync.Map{}
}

// --- init() registration ---

func init() {
	common.Register("wordpiece", func(data *common.TokenizerData) (common.Tokenizer, error) {
		return New(data)
	})
}
