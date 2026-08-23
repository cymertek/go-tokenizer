package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

// TestComprehensiveTokenization exercises all tokenization strategies against real GGUF files.
func TestComprehensiveTokenization(t *testing.T) {
	ggufFiles := []string{
		"/workdir/Bonsai-8B.gguf",
	}

	for _, ggufPath := range ggufFiles {
		if _, err := os.Stat(ggufPath); os.IsNotExist(err) {
			t.Skipf("Skipping %s: file not found", ggufPath)
		}

		t.Run(filepath.Base(ggufPath), func(t *testing.T) {
			testGGUFFile(t, ggufPath)
		})
	}
}

func testGGUFFile(t *testing.T, path string) {
	data, err := ReadTokenizerFromGGUF(path)
	if err != nil {
		t.Fatalf("ReadTokenizerFromGGUF(%q): %v", path, err)
	}

	if data.Model == "" {
		t.Error("Model should not be empty")
	}

	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	t.Logf("Model: %s, Tokens: %d, Merges: %d, BOS: %d, EOS: %d",
		data.Model, len(data.Tokens), len(data.Merges), data.BOSID, data.EOSID)

	// Test basic encoding/decoding
	testCases := []string{
		"Hello world",
		"Test string",
		"12345",
		"Special chars: @#$%",
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("encode_%s", strings.ReplaceAll(tc, " ", "_")), func(t *testing.T) {
			ids := bpe.EncodeIDs(tc)
			if len(ids) == 0 {
				t.Error("EncodeIDs returned empty slice")
				return
			}

			decoded := bpe.Detokenize(ids)
			t.Logf("Input: %q -> IDs: %v -> Decoded: %q", tc, ids, decoded)

			// Check that token count matches Count
			count := bpe.Count(tc)
			if count != len(ids) {
				t.Errorf("Count(%d) != len(EncodeIDs)(%d)", count, len(ids))
			}
		})
	}

	// Test Tokens() method
	t.Run("tokens_method", func(t *testing.T) {
		tokens := bpe.Tokens("Hello world")
		if len(tokens) == 0 {
			t.Error("Tokens() returned empty slice")
		} else {
			t.Logf("Tokens: %v", tokens)
		}
	})

	// Test HasToken and TokenID
	if len(data.Tokens) > 0 {
		t.Run("has_token", func(t *testing.T) {
			if !bpe.HasToken(data.Tokens[0]) {
				t.Errorf("HasToken(%q) should be true", data.Tokens[0])
			}
			id := bpe.TokenID(data.Tokens[0])
			if id < 0 || id >= len(data.Tokens) {
				t.Errorf("TokenID(%q) = %d, want valid ID in [0, %d)", data.Tokens[0], id, len(data.Tokens))
			}
		})
	}

	// Test cache functionality
	t.Run("cache", func(t *testing.T) {
		bpe.SetCache(10)
		ids1 := bpe.EncodeIDs("cached test")
		ids2 := bpe.EncodeIDs("cached test")
		if len(ids1) != len(ids2) {
			t.Error("Cached and uncached results should match")
		}

		bpe.ClearCache()
	})
}

// TestPreTypeStrategies verifies all pre-tokenization strategies are accessible.
func TestPreTypeStrategies(t *testing.T) {
	strategies := []struct {
		name  string
		pre   PreType
		model string
	}{
		{"gpt-2", PreGPT2, "gpt-2"},
		{"qwen2", PreQwen2, "qwen2"},
		{"llama3", PreLlama3, "llama3"},
		{"starcoder", PreStarcoder, "starcoder"},
		{"deepseek-llm", PreDeepSeekLLM, "deepseek-llm"},
		{"falcon", PreFalcon, "falcon"},
		{"qwen35", PreQwen35, "qwen35"},
		{"stablelm2", PreStableLM2, "stablelm2"},
		{"gpt-4o", PreGPT4O, "gpt-4o"},
		{"gemma4", PreGemma4, "gemma4"},
	}

	for _, s := range strategies {
		t.Run(s.name, func(t *testing.T) {
			pt := common.PreTypeFromString(s.model)
			if pt != s.pre {
				t.Errorf("PreTypeFromString(%q) = %d, want %d", s.model, pt, s.pre)
			}
		})
	}

	// Test default case
	t.Run("unknown_model", func(t *testing.T) {
		pt := common.PreTypeFromString("unknown-model")
		if pt != PreDefault {
			t.Errorf("PreTypeFromString(unknown) = %d, want PreDefault(%d)", pt, PreDefault)
		}
	})
}

// TestBPEModelDetection verifies model type detection.
func TestBPEModelDetection(t *testing.T) {
	tests := []struct {
		model  string
		expect string
	}{
		{"gpt-2", "bpe"},
		{"llama3", "bpe"},
		{"qwen2", "bpe"},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			data := &TokenizerData{Model: tt.model}
			bpe, err := NewBPE(data)
			if err != nil || bpe == nil {
				t.Fatalf("NewBPE error: %v", err)
			}
			if bpe.Type() != tt.expect {
				t.Errorf("Type() = %q, want %q", bpe.Type(), tt.expect)
			}
		})
	}
}
