// Package common provides shared types and interfaces used by all tokenizer
// sub-packages (bpe, spm, unigram, wordpiece). This avoids circular import
// dependencies — sub-packages can reference these types without importing their
// parent package.
package common

import "fmt"

// ---- Shared configuration ----

// TokenizerData holds all tokenizer metadata, whether extracted from a GGUF file
// or provided programmatically. Use Has* flags to distinguish "not set" from
// "set to zero" — the Go zero-value problem.
type TokenizerData struct {
	// Model identifies the tokenizer type: "bpe", "spm", "unigram", "wordpiece".
	// Required for programmatic construction; auto-detected from GGUF files.
	Model string

	// Tokens is the full vocabulary (token string → index). Each token's position
	// in this slice corresponds to its integer ID. Required for all tokenizer types.
	Tokens []string

	// Merges holds BPE merge rules as ordered pairs of string fragments. Only used
	// by BPE tokenizers. The order determines merge priority (earlier merges apply first).
	Merges []Merge

	// SPMModel holds the protobuf-encoded SentencePiece model binary. Populated when
	// tokenizer.ggml.spm_model is present in a GGUF file; nil for programmatic config.
	SPMModel []byte

	// WPModel holds WordPiece model binary (if any). Currently unused but reserved
	// for future WordPiece protobuf support.
	WPModel []byte

	// Special token IDs identify boundary markers:
	//   - BOSID/EOTID: beginning-of-sequence / end-of-text
	//   - EOSID/EOMID: end-of-sentence / end-of-message
	//   - UNKID: unknown token for out-of-vocabulary input
	//   - PADID: padding token for batch alignment
	// Each ID is paired with a Has* flag to distinguish "not set" (-1) from "set to zero".
	BOSID, EOSID, EOTID, EOMID, UNKID, PADID int64

	// HasBOSID through HasPADID indicate whether the corresponding ID field was explicitly
	// set. Use these flags instead of checking for -1 to correctly handle IDs that are zero.
	HasBOSID, HasEOSID, HasEOTID, HasEOMID, HasUNKID, HasPADID bool

	// AddBOS prepends the BOS token to encoded output; AddEOS appends the EOS token.
	// Both must be true for their respective tokens to appear in EncodeIDs results.
	AddBOS, AddEOS bool

	// PreType selects the pre-tokenization strategy for BPE models (e.g., GPT-2, Llama3).
	// Ignored by SPM/Unigram/WordPiece tokenizers which have their own segmentation rules.
	PreType PreType

	// TokenType holds per-token type flags extracted from GGUF's tokenizer.ggml.token_type.
	// Each entry corresponds to the token at the same index in Tokens. Currently unused.
	TokenType []int32

	// SpaceChar is the space prefix character detected from vocabulary (e.g., Ġ = U+0120).
	// Used by BPE models to mark word boundaries; set to 0 if not applicable.
	SpaceChar rune

	// SPMProbabilities holds unigram model probabilities as interleaved [tokenID, prob] pairs.
	// Only used by Unigram tokenizers for Viterbi segmentation. Length must be even.
	SPMProbabilities []float64

	// Config holds additional tokenizer-specific configuration (e.g., normalization rules).
	// Keys are model-type-specific; values may be strings, ints, bools, or nested maps.
	Config map[string]interface{}
}

// Merge holds a single BPE merge rule (pair of string fragments).
// Merges are applied in order: earlier merges take priority. The A and B fields
// represent the two string fragments to concatenate; they may be empty strings
// for special tokens or whitespace markers.
type Merge struct {
	// A is the left fragment in the merge pair (e.g., "hel", "▁").
	A string
	// B is the right fragment in the merge pair (e.g., "lo", "world").
	B string
}

// ---- Tokenizer interface ----

// Tokenizer is the common interface implemented by all tokenizer types.
// Each sub-package (bpe, spm, unigram, wordpiece) provides an implementation.
type Tokenizer interface {
	// EncodeIDs converts text into a slice of integer token IDs using greedy
	// longest-match or Viterbi segmentation depending on the model type.
	EncodeIDs(text string) []int

	// Detokenize converts a slice of token IDs back to human-readable text,
	// handling space markers (▁ → space), continuation prefixes (## → join),
	// and special token stripping ([CLS], [SEP], etc.).
	Detokenize(ids []int) string

	// Count returns the number of tokens that would be produced for the given text,
	// including BOS/EOS if configured. Equivalent to len(EncodeIDs(text)).
	Count(text string) int

	// Type returns a string identifier for this tokenizer's algorithm type:
	// "bpe", "spm", "unigram", or "wordpiece".
	Type() string

	// HasToken reports whether the given token string exists in the vocabulary.
	HasToken(tok string) bool

	// TokenID returns the integer ID for a token string, or -1 if not found.
	TokenID(tok string) int
}

// ---- Programmatic construction ----

// NewProgrammatic creates a TokenizerData struct populated with the given model
// type and vocabulary tokens. Use this for in-memory configuration without reading
// from a GGUF file. The optional merges parameter allows specifying BPE merge rules.
//
// Example:
//
//	data := common.NewProgrammatic("bpe", []string{"hello", "world"}, []common.Merge{{A: "hel", B: "lo"}})
func NewProgrammatic(model string, tokens []string, merges ...[]Merge) *TokenizerData {
	data := &TokenizerData{Model: model, Tokens: tokens}
	if len(merges) > 0 {
		data.Merges = merges[0]
	}
	return data
}

// New creates a tokenizer instance from the given TokenizerData by dispatching to
// the appropriate sub-package based on data.Model. The model type must be one of:
// "bpe", "spm", "unigram", "wordpiece". If data is nil, an error is returned.
//
// Example:
//
//	data := common.NewProgrammatic("spm", []string{"▁hello", "world"})
//	tok, err := common.New(data)
func New(data *TokenizerData) (Tokenizer, error) {
	if data == nil {
		return nil, fmt.Errorf("tokenizer: nil data")
	}

	constructor, ok := registry[data.Model]
	if !ok {
		// Fallback to BPE if model type unknown but tokens/merges present.
		if len(data.Tokens) > 0 && len(data.Merges) > 0 {
			data.Model = "bpe"
			constructor, ok = registry["bpe"]
			if !ok {
				return nil, fmt.Errorf("tokenizer: no constructor registered for model %q", data.Model)
			}
		} else {
			return nil, fmt.Errorf("tokenizer: unknown model type %q (registered types: %v)", data.Model, RegisteredTypes())
		}
	}

	tok, err := constructor(data)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: failed to create %s tokenizer: %w", data.Model, err)
	}

	return tok, nil
}
