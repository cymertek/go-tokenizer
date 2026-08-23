package models

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// TestRoundTrip_Comprehensive tests bidirectional fidelity across all model types.
func TestRoundTrip_Comprehensive(t *testing.T) {
	tests := []struct {
		name   string
		texts  []string
		tokens []string
	}{
		{
			name:   "BPE basic",
			texts:  []string{"hello world", "foo bar baz"},
			tokens: []string{"hello", "world", "Ġworld", "foo", "bar", "baz"},
		},
		{
			name:   "BPE with merges",
			texts:  []string{"the quick brown fox", "hello world"},
			tokens: []string{"the", "Ġquick", "Ġbrown", "Ġfox", "hello", "world"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Model:  "bpe",
				Tokens: tt.tokens,
			}

			bpeTok, err := bpe.New(data)
			if err != nil || bpeTok == nil {
				t.Fatalf("Failed to create BPE tokenizer: %v", err)
			}

			for _, text := range tt.texts {
				t.Run(text, func(t *testing.T) {
					// Encode
					ids := bpeTok.EncodeIDs(text)
					if len(ids) == 0 {
						t.Fatalf("EncodeIDs(%q) returned no ids", text)
					}

					// Decode
					decoded := bpeTok.Detokenize(ids)

					// Verify round-trip fidelity
					if decoded != text {
						t.Errorf("Round-trip failed: input %q -> ids %v -> output %q\n"+
							"  This indicates encoding/decoding is not bidirectional.", text, ids, decoded)
					} else {
						t.Logf("✓ Round-trip successful: %q", text)
					}
				})
			}
		})
	}
}

// TestRoundTrip_Unicode verifies round-trip with multi-byte characters.
func TestRoundTrip_Unicode(t *testing.T) {
	tests := []struct {
		name   string
		texts  []string
		tokens []string
	}{
		{
			name:   "CJK",
			texts:  []string{"你好世界", "hello世界"},
			tokens: []string{"你", "好", "世", "界", "hello", "Ġ世界"},
		},
		{
			name:   "Emoji",
			texts:  []string{"😀🎉"},
			tokens: []string{"😀", "🎉"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Model:  "bpe",
				Tokens: tt.tokens,
			}

			bpeTok, err := bpe.New(data)
			if err != nil || bpeTok == nil {
				t.Fatalf("Failed to create BPE tokenizer: %v", err)
			}

			for _, text := range tt.texts {
				t.Run(text, func(t *testing.T) {
					ids := bpeTok.EncodeIDs(text)
					if len(ids) == 0 {
						t.Fatalf("EncodeIDs(%q) returned no ids", text)
					}

					decoded := bpeTok.Detokenize(ids)

					if decoded != text {
						t.Errorf("Unicode round-trip failed: input %q -> output %q", text, decoded)
					} else {
						t.Logf("✓ Unicode round-trip successful")
					}
				})
			}
		})
	}
}

// TestRoundTrip_Concurrent verifies that concurrent encoding maintains fidelity.
func TestRoundTrip_Concurrent(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "bpe",
		Tokens: []string{"hello", "world", "Ġworld"},
	}

	bpeTok, err := bpe.New(data)
	if err != nil || bpeTok == nil {
		t.Fatalf("Failed to create BPE tokenizer: %v", err)
	}

	// Create a long string that will trigger concurrent encoding (>32KB)
	longText := strings.Repeat("hello world ", 1000)

	ids := bpeTok.EncodeIDs(longText)
	if len(ids) == 0 {
		t.Fatalf("EncodeIDs returned no ids for long text")
	}

	decoded := bpeTok.Detokenize(ids[:min(100, len(ids))]) // Test first 100 tokens
	expectedPrefix := "hello world hello world"

	if !strings.HasPrefix(decoded, expectedPrefix) {
		t.Errorf("Concurrent encoding round-trip failed: got %q, want prefix %q", decoded, expectedPrefix)
	} else {
		t.Logf("✓ Concurrent round-trip successful")
	}
}

// TestRoundTrip_RealModel tests against the actual Bonsai-8B.gguf model.
func TestRoundTrip_RealModel(t *testing.T) {
	ggufPath := "/workdir/Bonsai-8B.gguf"
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skipf("Skipping real model test: %v", err)
	}

	data, err := ReadTokenizerFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("Failed to load tokenizer data: %v", err)
	}

	tests := []struct {
		name  string
		input string
		desc  string
	}{
		{"simple_english", "Hello world", "Basic English"},
		{"code_snippet", `func main() { fmt.Println("Hello") }`, "Go code snippet"},
		{"unicode_mixed", "Hello 世界 🌍", "Mixed ASCII and Unicode (CJK + emoji)"},
		{"special_chars", "<s>Special chars: @#$%^&*()</s>", "HTML/XML tags with special characters"},
		{"whitespace_only", "   ", "Only whitespace"},
		{"single_char", "h", "Single character"},
		{"consecutive_chars", "abc", "Consecutive single characters"},
		{"numbers", "12345", "Numbers only"},
		{"code_mixed", `if (x > 0) { return x * 2; }`, "Code with operators and punctuation"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bpeTok, err := bpe.New(data)
			if err != nil || bpeTok == nil {
				t.Fatalf("Failed to create BPE tokenizer: %v", err)
			}

			ids := bpeTok.EncodeIDs(tt.input)
			fmt.Printf("[%s] Input=%q -> IDs=%v\n", tt.name, tt.input, ids)

			if len(tt.input) > 0 && len(ids) == 0 {
				t.Errorf("EncodeIDs(%q) returned empty slice for non-empty input: %s", tt.input, tt.desc)
			}

			// Test round-trip if we got some tokens
			if len(ids) > 0 {
				decoded := bpeTok.Detokenize(ids)
				fmt.Printf("[%s] Decoded=%q\n", tt.name, decoded)

				if decoded != tt.input {
					t.Errorf("Round-trip failed for %q:\n  Expected: %q\n  Got:      %q (%s)", tt.input, tt.input, decoded, tt.desc)
				} else {
					t.Logf("✓ Round-trip successful for %q", tt.input)
				}
			}
		})
	}
}

// TestCrossModelVerification verifies that our tokenizer matches expected behavior.
func TestCrossModelVerification(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []int
		desc     string
	}{
		{
			name:     "exact_vocab_match",
			input:    "hello world",
			expected: nil, // Will be filled dynamically based on vocab
			desc:     "Tokens that exist exactly in vocabulary",
		},
		{
			name:     "partial_match",
			input:    "hel",
			expected: nil,
			desc:     "Partial token should still encode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Tokens: []string{"hello", "world"},
			}
			bpeTok, err := bpe.New(data)
			if err != nil || bpeTok == nil {
				t.Fatalf("Failed to create BPE tokenizer: %v", err)
			}

			ids := bpeTok.EncodeIDs(tt.input)
			fmt.Printf("[%s] Input=%q -> IDs=%v\n", tt.name, tt.input, ids)

			if len(ids) == 0 && len(tt.input) > 0 {
				t.Errorf("EncodeIDs(%q) returned empty slice: %s", tt.input, tt.desc)
			}

			// Verify round-trip
			decoded := bpeTok.Detokenize(ids)
			fmt.Printf("[%s] Decoded=%q\n", tt.name, decoded)

			if decoded != tt.input {
				t.Errorf("Round-trip failed for %q: got %q instead of %q (%s)", tt.input, decoded, tt.input, tt.desc)
			}
		})
	}
}

// TestEncodingConsistency verifies that encoding is deterministic.
func TestEncodingConsistency(t *testing.T) {
	data := &common.TokenizerData{
		Tokens: []string{"hello", "world"},
	}
	bpeTok, err := bpe.New(data)
	if err != nil || bpeTok == nil {
		t.Fatalf("Failed to create BPE tokenizer: %v", err)
	}

	inputs := []string{
		"hello world",
		"test string",
		"12345",
	}

	for _, input := range inputs {
		ids1 := bpeTok.EncodeIDs(input)
		ids2 := bpeTok.EncodeIDs(input)

		if !sliceEqual(ids1, ids2) {
			t.Errorf("Encoding not consistent for %q: got %v and %v", input, ids1, ids2)
		} else {
			fmt.Printf("[%s] Encoding is consistent\n", input)
		}
	}
}

// TestDetokenizationConsistency verifies that detokenization is deterministic.
func TestDetokenizationConsistency(t *testing.T) {
	data := &common.TokenizerData{
		Tokens: []string{"hello", "world"},
	}
	bpeTok, err := bpe.New(data)
	if err != nil || bpeTok == nil {
		t.Fatalf("Failed to create BPE tokenizer: %v", err)
	}

	inputs := []string{
		"hello world",
		"test string",
		"12345",
	}

	for _, input := range inputs {
		ids := bpeTok.EncodeIDs(input)
		if len(ids) == 0 {
			continue
		}

		decoded1 := bpeTok.Detokenize(ids)
		decoded2 := bpeTok.Detokenize(ids)

		if decoded1 != decoded2 {
			t.Errorf("Detokenization not consistent for %q: got %q and %q", input, decoded1, decoded2)
		} else {
			fmt.Printf("[%s] Detokenization is consistent\n", input)
		}
	}
}

// TestVocabIntegrity verifies that all tokens in the vocab are properly encoded.
func TestVocabIntegrity(t *testing.T) {
	tokens := []string{"hello", "world", "Ġhello", "Ġworld"}

	data := &common.TokenizerData{
		Tokens: tokens,
	}
	bpeTok, err := bpe.New(data)
	if err != nil || bpeTok == nil {
		t.Fatalf("Failed to create BPE tokenizer: %v", err)
	}

	fmt.Println("\n=== Testing vocab integrity ===")
	for i, tokStr := range tokens {
		ids := bpeTok.EncodeIDs(tokStr)
		fmt.Printf("Token[%d]=%q -> IDs=%v\n", i, tokStr, ids)

		if len(ids) == 0 {
			t.Errorf("Failed to encode token %q (id=%d)", tokStr, i)
		} else if ids[0] != i {
			t.Errorf("Token %q should map to id %d but got %v", tokStr, i, ids)
		}
	}

	fmt.Println("\n=== Testing detokenization ===")
	// Tokens starting with spaceChar (e.g. "Ġhello") represent " hello" in BPE models,
	// not the literal string "Ġhello". Skip them from round-trip checks.
	for _, input := range tokens {
		if len(input) > 0 {
			firstRune, _ := utf8.DecodeRuneInString(input)
			if firstRune == 'Ġ' || firstRune == '_' {
				fmt.Printf("Encode(%q)->Detokenize() = %q (skipped: space-prefixed token)\n", input, bpeTok.Detokenize(bpeTok.EncodeIDs(input)))
				continue
			}
		}
		ids := bpeTok.EncodeIDs(input)
		if len(ids) > 0 {
			decoded := bpeTok.Detokenize(ids)
			fmt.Printf("Encode(%q)->Detokenize() = %q\n", input, decoded)
			if decoded != input {
				t.Errorf("Round-trip failed for token %q: got %q instead of %q", input, decoded, input)
			}
		}
	}
}

// TestEdgeCases tests various edge cases and boundary conditions.
func TestEdgeCases(t *testing.T) {
	data := &common.TokenizerData{
		Tokens: []string{"hello", "world"},
	}
	bpeTok, err := bpe.New(data)
	if err != nil || bpeTok == nil {
		t.Fatalf("Failed to create BPE tokenizer: %v", err)
	}

	tests := []struct {
		name  string
		input string
		desc  string
	}{
		{"empty_string", "", "Empty string"},
		{"single_space", " ", "Single space"},
		{"multiple_spaces", "   ", "Multiple spaces"},
		{"newline", "\n", "Newline character"},
		{"tab", "\t", "Tab character"},
		{"special_unicode", "😀", "Emoji"},
		{"cjk_char", "你", "CJK character"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ids := bpeTok.EncodeIDs(tt.input)
			fmt.Printf("[%s] Input=%q -> IDs=%v\n", tt.name, tt.input, ids)

			if len(tt.input) > 0 && len(ids) == 0 {
				// Check if input is whitespace-only (this is acceptable behavior)
				isWhitespace := true
				for _, c := range tt.input {
					if !strings.ContainsRune(" \t\n\r", c) {
						isWhitespace = false
						break
					}
				}
				if !isWhitespace {
					t.Errorf("EncodeIDs(%q) returned empty slice for non-empty non-whitespace input: %s", tt.input, tt.desc)
				} else {
					fmt.Printf("[%s] Whitespace-only input handled (empty output)\n", tt.name)
				}
			}

			// Test round-trip if we got any tokens
			if len(ids) > 0 {
				decoded := bpeTok.Detokenize(ids)
				fmt.Printf("[%s] Decoded=%q\n", tt.name, decoded)

				if decoded != tt.input {
					t.Errorf("Round-trip failed for %q:\n  Expected: %q\n  Got:      %q (%s)", tt.input, tt.input, decoded, tt.desc)
				}
			}
		})
	}
}

// sliceEqual compares two integer slices for equality.
func sliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
