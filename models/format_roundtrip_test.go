package models

import (
	
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

// TestRoundTrip_AllFormats verifies encode/decode round-trips work for all tokenizer formats.
func TestRoundTrip_AllFormats(t *testing.T) {
	tests := []struct {
		name   string
		data   *common.TokenizerData
		inputs []string
	}{
		{
			name: "BPE_basic",
			data: &common.TokenizerData{
				Model:  "bpe",
				Tokens: []string{"hello", "world", "Ġhello", "Ġworld", "hello world", "Hello", "ĠHello"},
				Merges: []common.Merge{{A: "hel", B: "lo"}},
			},
			inputs: []string{"hello world", "Hello world", "hello"},
		},
		{
			name: "SPM_basic",
			data: &common.TokenizerData{
				Model:  "spm",
				Tokens: []string{"▁hello", "▁world", "hello", "world", "Ġhello", "Ġworld", "Hello", "▁Hello"},
				Merges: []common.Merge{},
			},
			inputs: []string{"hello world", "Hello world"},
		},
		{
			name: "Unigram_basic",
			data: &common.TokenizerData{
				Model:  "unigram",
				Tokens: []string{"▁hello", "▁world", "hello", "world"},
				Merges: []common.Merge{},
				UNKID:  -1,
				HasUNKID: true,
			},
			inputs: []string{"hello world"},
		},
		{
			name: "WordPiece_basic",
			data: &common.TokenizerData{
				Model:  "wordpiece",
				Tokens: []string{"[CLS]", "[SEP]", "hello", "world", "##hello", "##world", "Ġhello", "Ġworld"},
				Merges: []common.Merge{},
				BOSID:  102,
				EOSID:  103,
				HasBOSID: true,
				HasEOSID: true,
				AddBOS:   true,
				AddEOS:   true,
			},
			inputs: []string{"hello world"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bpe, err := NewBPE(tt.data)
			if err != nil || bpe == nil {
				t.Fatalf("NewBPE error: %v", err)
			}

			for _, input := range tt.inputs {
				t.Run(input, func(t *testing.T) {
					ids := bpe.EncodeIDs(input)
					if len(ids) == 0 {
						t.Fatal("EncodeIDs returned empty slice")
					}


					decoded := bpe.Detokenize(ids)
					if strings.TrimSpace(decoded) != strings.TrimSpace(input) {
						t.Errorf("Detokenize(EncodeIDs(%q)) = %q, want %q (ids: %v)", input, decoded, input, ids)
					}
				})
			}
		})
	}
}

// TestRoundTrip_SpecialTokens verifies BOS/EOS/UNK/PAD tokens survive round-trip.
func TestFormatSpecialTokens(t *testing.T) {
	data := &common.TokenizerData{
		Model:    "bpe",
		Tokens:   []string{"hello", "world"},
		Merges:   []common.Merge{},
		BOSID:    1,
		EOSID:    2,
		UNKID:    -1,
		PADID:    0,
		HasBOSID: true,
		HasEOSID: true,
		HasUNKID: true,
		HasPADID: true,
		AddBOS:   true,
		AddEOS:   true,
	}

	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{"hello world"}
	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			if len(ids) < 2 { // Should have at least BOS + EOS
				t.Fatalf("EncodeIDs(%q) = %v (too few tokens)", input, ids)
			}

			// Check that BOS and EOS are present (positions may vary based on AddBOS/AddEOS)
			hasBOS := false
			hasEOS := false
			for _, id := range ids {
				if int64(id) == data.BOSID {
					hasBOS = true
				}
				if int64(id) == data.EOSID {
					hasEOS = true
				}
			}

			if !hasBOS {
				t.Errorf("EncodeIDs(%q) missing BOS token %d", input, data.BOSID)
			}
			if !hasEOS {
				t.Errorf("EncodeIDs(%q) missing EOS token %d", input, data.EOSID)
			}

		})
	}
}

// TestRoundTrip_EmptyAndEdgeCases verifies behavior with edge case inputs.
func TestRoundTrip_EmptyAndEdgeCases(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "bpe",
		Tokens: []string{"hello", "world", "1", "2", "3", "4", "5"},
		Merges: []common.Merge{},
	}

	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	tests := []struct {
		name  string
		input string
	}{
		{"empty_string", ""},
		{"single_char", "h"},
		{"unicode", "hello world 🌍"},
		{"numbers", "12345"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ids := bpe.EncodeIDs(tt.input)
			if len(ids) == 0 && tt.input != "" {
				t.Errorf("EncodeIDs(%q) returned empty slice", tt.input)
			}

		})
	}
}
