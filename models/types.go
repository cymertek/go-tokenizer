// Package models provides GGUF tokenizer support: extract vocabularies and merges
// from GGUF files and produce correct tokenization across multiple algorithm types.
//
// This package reads GGUF KV pairs, builds tokenizer instances (BPE, SPM, Unigram,
// WordPiece), and dispatches to the appropriate implementation based on the model
// type declared in the GGUF file.
//
// Each tokenizer sub-package (bpe/, spm/, unigram/, wordpiece/) registers itself
// via init() into a central registry. New tokenizers can be added by creating a
// new sub-package that implements the Tokenizer interface and calls Register().
package models

import (
	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// Re-exports from common for backwards compatibility — users can still reference
// these types via the root gguf package.
type (
	TokenizerData = common.TokenizerData
	Merge         = common.Merge
	PreType       = common.PreType
	Constructor   = common.Constructor
	Tokenizer     = common.Tokenizer
)

// BPE re-export for backwards compatibility with existing tests and examples.
// New code should import github.com/cymertek/go-tokenizer/models/bpe directly.
type BPE = bpe.BPE

// NewBPE is a convenience wrapper around bpe.New() for backward compatibility.
func NewBPE(data *TokenizerData) (*BPE, error) {
	return bpe.New(data)
}

const (
	PreDefault     = common.PreDefault
	PreGPT2        = common.PreGPT2
	PreQwen2       = common.PreQwen2
	PreLlama3      = common.PreLlama3
	PreStarcoder   = common.PreStarcoder
	PreDeepSeekLLM = common.PreDeepSeekLLM
	PreFalcon      = common.PreFalcon
	PreQwen35      = common.PreQwen35
	PreStableLM2   = common.PreStableLM2
	PreGPT4O       = common.PreGPT4O
	PreGemma4      = common.PreGemma4
)

// NewProgrammatic creates a TokenizerData for in-memory configuration without a GGUF file.
func NewProgrammatic(model string, tokens []string, merges ...[]common.Merge) *TokenizerData {
	return common.NewProgrammatic(model, tokens, merges...)
}

// New creates a tokenizer from extracted or programmatic data. It dispatches to the
// appropriate sub-package based on data.Model using the auto-registration registry.
func New(data *TokenizerData) (Tokenizer, error) {
	return common.New(data)
}

// Register adds a tokenizer constructor for the given model type.
// Called by init() in each sub-package during package initialization.
func Register(modelType string, fn Constructor) {
	common.Register(modelType, fn)
}

// RegisteredTypes returns the list of all registered tokenizer types.
func RegisteredTypes() []string {
	return common.RegisteredTypes()
}
