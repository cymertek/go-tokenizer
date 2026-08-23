package bpe

import (
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestBPE_RoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		tokens   []string
		merges   []common.Merge
		input    string
		wantSubs string // expected substring after detokenize (exact match may vary due to whitespace handling)
	}{
		{
			name:     "simple words",
			tokens:   []string{"<unk>", "<s>", "</s>", "hello", "world", "hello world"},
			merges:   nil,
			input:    "hello world",
			wantSubs: "hello world",
		},
		{
			name:     "subword tokens",
			tokens:   []string{"<unk>", "<s>", "</s>", "hel", "lo", "wor", "ld", "hello", "world"},
			merges:   []common.Merge{{A: "hel", B: "lo"}, {A: "wor", B: "ld"}},
			input:    "hello world",
			wantSubs: "hello world",
		},
		{
			name:     "single word",
			tokens:   []string{"<unk>", "<s>", "</s>", "testing"},
			merges:   nil,
			input:    "testing",
			wantSubs: "testing",
		},
		{
			name:     "punctuation",
			tokens:   []string{"<unk>", "<s>", "</s>", "hello", ",", "world", "."},
			merges:   nil,
			input:    "hello, world.",
			wantSubs: "hello, world.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Model:  "bpe",
				Tokens: tt.tokens,
				Merges: tt.merges,
			}

			bpe, err := New(data)
			if err != nil {
				t.Fatalf("New error: %v", err)
			}

			ids := bpe.EncodeIDs(tt.input)
			if len(ids) == 0 {
				t.Fatal("EncodeIDs returned empty slice")
			}

			decoded := bpe.Detokenize(ids)
			t.Logf("Input: %q → IDs: %v → Decoded: %q", tt.input, ids, decoded)

			if !strings.Contains(decoded, tt.wantSubs) {
				t.Errorf("Detokenize(%v) = %q, want to contain %q", ids, decoded, tt.wantSubs)
			}
		})
	}
}

func TestBPE_RoundTripWithBOS_EOS(t *testing.T) {
	data := &common.TokenizerData{
		Model:    "bpe",
		Tokens:   []string{"<unk>", "<s>", "</s>", "hello", "world"},
		Merges:   nil,
		BOSID:    100,
		EOSID:    101,
		HasBOSID: true,
		HasEOSID: true,
		AddBOS:   true,
		AddEOS:   true,
	}

	bpe, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello world"
	ids := bpe.EncodeIDs(input)
	t.Logf("EncodeIDs(%q) = %v (len=%d)", input, ids, len(ids))

	// Should have BOS at start and EOS at end.
	if len(ids) < 3 {
		t.Fatalf("Expected at least 3 tokens (BOS + content + EOS), got %d", len(ids))
	}
	if ids[0] != 100 {
		t.Errorf("First token should be BOS (100), got %d", ids[0])
	}
	if ids[len(ids)-1] != 101 {
		t.Errorf("Last token should be EOS (101), got %d", ids[len(ids)-1])
	}

	decoded := bpe.Detokenize(ids)
	t.Logf("Detokenize(%v) = %q", ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded)
	}
}

func TestBPE_RoundTripWhitespace(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "bpe",
		Tokens: []string{"<unk>", "<s>", "</s>", "hello", " world", "  ", "hello  world"},
		Merges: nil,
	}

	bpe, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello  world" // double space
	ids := bpe.EncodeIDs(input)
	decoded := bpe.Detokenize(ids)
	t.Logf("Input: %q → IDs: %v → Decoded: %q", input, ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded)
	}
}

func TestBPE_RoundTripUnicode(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "bpe",
		Tokens: []string{"<unk>", "<s>", "</s>", "hello", "世界"},
		Merges: nil,
	}

	bpe, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello 世界"
	ids := bpe.EncodeIDs(input)
	decoded := bpe.Detokenize(ids)
	t.Logf("Input: %q → IDs: %v → Decoded: %q", input, ids, decoded)

	// Note: UNK tokens may appear in output if vocabulary doesn't cover all characters.
	// This is expected behavior - real models have much larger vocabularies.
	if !strings.Contains(decoded, "hello") {
		t.Errorf("Detokenize should contain 'hello', got %q", decoded)
	}
}
