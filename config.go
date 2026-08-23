package tokenizer

import (
	"encoding/json"
	"os"
)

// Config construct configuration for creating Tokenizer.
type Config struct {
	Version       string                 `json:"version"`
	Truncation    map[string]any         `json:"truncation"`
	Padding       map[string]any         `json:"padding"`
	AddedTokens   []TokenConfig          `json:"added_tokens"`
	Normalizer    map[string]any         `json:"normalizer"`
	PreTokenizer  map[string]any         `json:"pre_tokenizer"`
	PostProcessor map[string]any         `json:"post_processor"`
	Decoder       map[string]any         `json:"decoder"`
	Model         map[string]any         `json:"model"`
}

// TokenConfig holds configuration for a token added to the vocabulary.
type TokenConfig struct {
	Id         int64  `json:"id"`          //nolint:revive // matches HuggingFace JSON convention
	Content    string `json:"content"`     // The token content string
	SingleWord bool   `json:"single_word"` // Whether the token must match as a whole word
	Lstrip     bool   `json:"lstrip"`      // Strip leading whitespace when matching
	Rstrip     bool   `json:"rstrip"`      // Strip trailing whitespace when matching
	Normalized bool   `json:"normalized"`  // Whether to normalize before matching
	Special    bool   `json:"special"`     // Whether this is a special token
}

// NormalizerConfig describes the normalizer chain used during preprocessing.
type NormalizerConfig struct {
	Type        string               `json:"type"`        // The normalizer type name (e.g., "bert", "unicode").
	Normalizers []map[string]any     `json:"normalizers"` // Ordered list of normalizer configurations.
}

// PreTokenizerConfig describes pre-tokenization configuration.
type PreTokenizerConfig struct{}

// PostProcessorConfig describes post-processing configuration including pair handling.
type PostProcessorConfig struct {
	Type          string               `json:"type"`           // The processor type (e.g., "bert", "template").
	Single        []map[string]any     `json:"single"`         // Configuration for single-sequence processing.
	Pair          []map[string]any     `json:"pair"`           // Configuration for pair-sequence processing.
	SpecialTokens map[string]any       `json:"speical_tokens"` // Special token handling configuration (note: "speical" matches HF convention).
}

// DecoderConfig describes the decoder chain used during detokenization.
type DecoderConfig struct {
	Type     string               `json:"type"`      // The decoder type name (e.g., "byte_fallback", "ctc").
	Decoders []map[string]any     `json:"decoders"`  // Ordered list of decoder configurations.
}

// ModelConfig describes model-specific tokenization configuration.
type ModelConfig struct {
	Type                    string         `json:"type"`                     // The model type (e.g., "BPE", "WordPiece").
	Dropout                 any            `json:"dropout"`                  // Dropout probability for unigram models.
	UnkToken                string         `json:"unk_token"`                // The unknown token string.
	ContinuingSubwordPrefix any            `json:"continuing_subword_prefix"`// Prefix for subwords that continue a word (WordPiece).
	EndOfWordSuffix         any            `json:"end_of_word_suffix"`       // Suffix marking end of word (WordPiece).
	FuseUnk                 bool           `json:"fuse_unk"`                 // Whether to fuse [UNK] with subsequent tokens.
	ByteFallback            bool           `json:"byte_fallback"`            // Whether to fall back to byte-level encoding.
	Vocab                   map[string]int `json:"vocab"`                    // Token string → ID mapping.
	Merges                  []string       `json:"merges"`                   // BPE merge rules as "a b" strings.
	MaxInputCharsPerWord    any            `json:"max_input_chars_per_word"` // Maximum characters per word for unigram training.
}

// ConfigFromFile loads config from file.
func ConfigFromFile(file string) (*Config, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}

	dec := json.NewDecoder(f)

	var config *Config
	err = dec.Decode(&config)
	if err != nil {
		return nil, err
	}

	return config, nil
}

// Vocab is the vocabulary mapping token string to token ID.
type Vocab map[string]int

// VocabR is the reversed vocabulary mapping token ID to token string.
type VocabR map[int]string
