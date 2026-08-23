package spm

import (
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestSPM_RoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		tokens   []string
		input    string
		wantSubs string
	}{
		{
			name:     "simple words",
			tokens:   []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
			input:    "hello world",
			wantSubs: "hello world",
		},
		{
			name:     "single word with meta",
			tokens:   []string{"<unk>", "<s>", "</s>", "▁testing"},
			input:    "testing",
			wantSubs: "testing",
		},
		{
			name:     "multiple words",
			tokens:   []string{"<unk>", "<s>", "</s>", "▁hello", "▁world", "▁foo"},
			input:    "hello world foo",
			wantSubs: "hello world foo",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Model:  "spm",
				Tokens: tt.tokens,
			}

			spm, err := New(data)
			if err != nil {
				t.Fatalf("New error: %v", err)
			}

			ids := spm.EncodeIDs(tt.input)
			if len(ids) == 0 {
				t.Fatal("EncodeIDs returned empty slice")
			}

			decoded := spm.Detokenize(ids)
			t.Logf("Input: %q → IDs: %v → Decoded: %q", tt.input, ids, decoded)

			if !strings.Contains(decoded, tt.wantSubs) {
				t.Errorf("Detokenize(%v) = %q, want to contain %q", ids, decoded, tt.wantSubs)
			}
		})
	}
}

func TestSPM_RoundTripWithBOS_EOS(t *testing.T) {
	data := &common.TokenizerData{
		Model:    "spm",
		Tokens:   []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
		BOSID:    100,
		EOSID:    101,
		HasBOSID: true,
		HasEOSID: true,
		AddBOS:   true,
		AddEOS:   true,
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello world"
	ids := spm.EncodeIDs(input)
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

	decoded := spm.Detokenize(ids)
	t.Logf("Detokenize(%v) = %q", ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded)
	}
}

func TestSPM_RoundTripUnicode(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁世界"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello 世界"
	ids := spm.EncodeIDs(input)
	decoded := spm.Detokenize(ids)
	t.Logf("Input: %q → IDs: %v → Decoded: %q", input, ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "世界") {
		t.Errorf("Detokenize should contain 'hello' and '世界', got %q", decoded)
	}
}

func TestSPM_RoundTripEmpty(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := ""
	ids := spm.EncodeIDs(input)
	if len(ids) != 0 {
		t.Errorf("EncodeIDs(%q) = %v, want empty slice", input, ids)
	}

	decoded := spm.Detokenize(ids)
	if decoded != "" {
		t.Errorf("Detokenize(%v) = %q, want empty string", ids, decoded)
	}
}
