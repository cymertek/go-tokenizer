package wordpiece

import (
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestWordPiece_RoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		tokens   []string
		input    string
		wantSubs string
	}{
		{
			name:     "simple words",
			tokens:   []string{"[UNK]", "[CLS]", "[SEP]", "hello", "world"},
			input:    "hello world",
			wantSubs: "hello world",
		},
		{
			name:     "subword with continuation",
			tokens:   []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing"},
			input:    "playing",
			wantSubs: "playing",
		},
		{
			name:     "multiple subwords",
			tokens:   []string{"[UNK]", "[CLS]", "[SEP]", "hello", "world", "foo", "##bar"},
			input:    "hello world foobar",
			wantSubs: "hello world foobar",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Model:  "wordpiece",
				Tokens: tt.tokens,
			}

			w, err := New(data)
			if err != nil {
				t.Fatalf("New error: %v", err)
			}

			ids := w.EncodeIDs(tt.input)
			if len(ids) == 0 {
				t.Fatal("EncodeIDs returned empty slice")
			}

			decoded := w.Detokenize(ids)
			t.Logf("Input: %q → IDs: %v → Decoded: %q", tt.input, ids, decoded)

			if !strings.Contains(decoded, tt.wantSubs) {
				t.Errorf("Detokenize(%v) = %q, want to contain %q", ids, decoded, tt.wantSubs)
			}
		})
	}
}

func TestWordPiece_RoundTripWithBOS_EOS(t *testing.T) {
	data := &common.TokenizerData{
		Model:    "wordpiece",
		Tokens:   []string{"[UNK]", "[CLS]", "[SEP]", "hello", "world"},
		BOSID:    1,
		EOSID:    2,
		HasBOSID: true,
		HasEOSID: true,
		AddBOS:   true,
		AddEOS:   true,
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello world"
	ids := w.EncodeIDs(input)
	t.Logf("EncodeIDs(%q) = %v (len=%d)", input, ids, len(ids))

	// Should have BOS at start and EOS at end.
	if len(ids) < 3 {
		t.Fatalf("Expected at least 3 tokens (BOS + content + EOS), got %d", len(ids))
	}
	if ids[0] != 1 {
		t.Errorf("First token should be BOS (1), got %d", ids[0])
	}
	if ids[len(ids)-1] != 2 {
		t.Errorf("Last token should be EOS (2), got %d", ids[len(ids)-1])
	}

	decoded := w.Detokenize(ids)
	t.Logf("Detokenize(%v) = %q", ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded)
	}
}

func TestWordPiece_RoundTripUnicode(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "hello", "世", "##界"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello 世界"
	ids := w.EncodeIDs(input)
	decoded := w.Detokenize(ids)
	t.Logf("Input: %q → IDs: %v → Decoded: %q", input, ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "世界") {
		t.Errorf("Detokenize should contain 'hello' and '世界', got %q", decoded)
	}
}

func TestWordPiece_RoundTripEmpty(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := ""
	ids := w.EncodeIDs(input)
	if len(ids) != 0 {
		t.Errorf("EncodeIDs(%q) = %v, want empty slice", input, ids)
	}

	decoded := w.Detokenize(ids)
	if decoded != "" {
		t.Errorf("Detokenize(%v) = %q, want empty string", ids, decoded)
	}
}
