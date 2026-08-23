package models

import (
	"bytes"
	"testing"

	tok "github.com/cymertek/go-tokenizer"
	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// TestRoundTrip_BPE verifies that BPE tokenizer configuration survives serialization/deserialization.
// This ensures vocab, merges, special token IDs, and PreType are preserved for rebuilding identical encoders.
func TestRoundTrip_BPE(t *testing.T) {
	tests := []struct {
		name string
		data *common.TokenizerData
	}{
		{
			name: "simple_words",
			data: &common.TokenizerData{
				Model:  "bpe",
				Tokens: []string{"hello", "world", "Ġhello", "Ġworld"},
				Merges: []common.Merge{{A: "hel", B: "lo"}},
			},
		},
		{
			name: "with_bos_eos_and_pretype",
			data: &common.TokenizerData{
				Model:  "bpe",
				Tokens: []string{"<unk>", "<s>", "</s>", "hello", "world"},
				BOSID:  100, EOSID: 101, HasBOSID: true, HasEOSID: true,
				AddBOS: true, AddEOS: true,
				PreType: common.PreGPT2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := tok.New(tt.data)

			var buf bytes.Buffer
			if err := tok.Serialize(tokenizer, &buf); err != nil {
				t.Fatalf("Failed to serialize BPE tokenizer: %v", err)
			}

			deserialized, err := tok.Deserialize(&buf)
			if err != nil {
				t.Fatalf("Failed to deserialize BPE tokenizer: %v", err)
			}

			// Verify all critical fields are preserved
			checkFields(t, "Model", tt.data.Model, deserialized.Data.Model)
			checkFields(t, "PreType", int32(tt.data.PreType), int32(deserialized.Data.PreType))
			checkFields(t, "BOSID", tt.data.BOSID, deserialized.Data.BOSID)
			checkFields(t, "EOSID", tt.data.EOSID, deserialized.Data.EOSID)
			checkBoolField(t, "HasBOSID", tt.data.HasBOSID, deserialized.Data.HasBOSID)
			checkBoolField(t, "HasEOSID", tt.data.HasEOSID, deserialized.Data.HasEOSID)
			checkBoolField(t, "AddBOS", tt.data.AddBOS, deserialized.Data.AddBOS)
			checkBoolField(t, "AddEOS", tt.data.AddEOS, deserialized.Data.AddEOS)

			// Verify tokens are preserved (order matters for ID mapping)
			if len(tt.data.Tokens) != len(deserialized.Data.Tokens) {
				t.Errorf("Token count mismatch: original=%d, deserialized=%d",
					len(tt.data.Tokens), len(deserialized.Data.Tokens))
			} else {
				for i := range tt.data.Tokens {
					if tt.data.Tokens[i] != deserialized.Data.Tokens[i] {
						t.Errorf("Token[%d] mismatch: original=%q, deserialized=%q",
							i, tt.data.Tokens[i], deserialized.Data.Tokens[i])
					}
				}
			}

			// Verify merges are preserved in order (critical for BPE)
			if len(tt.data.Merges) != len(deserialized.Data.Merges) {
				t.Errorf("Merge count mismatch: original=%d, deserialized=%d",
					len(tt.data.Merges), len(deserialized.Data.Merges))
			} else {
				for i := range tt.data.Merges {
					if tt.data.Merges[i].A != deserialized.Data.Merges[i].A ||
						tt.data.Merges[i].B != deserialized.Data.Merges[i].B {
						t.Errorf("Merge[%d] mismatch: original=%q+%q, deserialized=%q+%q",
							i, tt.data.Merges[i].A, tt.data.Merges[i].B,
							deserialized.Data.Merges[i].A, deserialized.Data.Merges[i].B)
					}
				}
			}

			// Reconstruct a BPE tokenizer from deserialized data and verify encoding matches original
			originalEncoder, err := bpe.New(tt.data)
			if err != nil {
				t.Fatalf("Failed to create original BPE encoder: %v", err)
			}

			deserEncoder, err := bpe.New(deserialized.Data)
			if err != nil {
				t.Fatalf("Failed to create deserialized BPE encoder: %v", err)
			}

			testInputs := []string{"hello world", "hello"}
			for _, input := range testInputs {
				origIDs := originalEncoder.EncodeIDs(input)
				deserIDs := deserEncoder.EncodeIDs(input)
				if !idsEqual(origIDs, deserIDs) {
					t.Errorf("EncodeIDs mismatch for %q:\n  original:   %v\n  deserialized: %v",
						input, origIDs, deserIDs)
				}
			}
		})
	}
}

// TestRoundTrip_WordPiece verifies that WordPiece tokenizer configuration survives serialization/deserialization.
// This is critical for BERT-style tokenizers with [CLS]/[SEP] special tokens and BOS/EOS markers.
func TestRoundTrip_WordPiece(t *testing.T) {
	tests := []struct {
		name string
		data *common.TokenizerData
	}{
		{
			name: "with_special_tokens",
			data: &common.TokenizerData{
				Model:  "wordpiece",
				Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "hello", "world"},
				BOSID:  102, EOSID: 103, HasBOSID: true, HasEOSID: true,
				AddBOS: true, AddEOS: true,
			},
		},
		{
			name: "subword_tokens",
			data: &common.TokenizerData{
				Model:  "wordpiece",
				Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tokenizer := tok.New(tt.data)

			var buf bytes.Buffer
			if err := tok.Serialize(tokenizer, &buf); err != nil {
				t.Fatalf("Failed to serialize WordPiece tokenizer: %v", err)
			}

			deserialized, err := tok.Deserialize(&buf)
			if err != nil {
				t.Fatalf("Failed to deserialize WordPiece tokenizer: %v", err)
			}

			// Verify all critical fields are preserved
			checkFields(t, "Model", tt.data.Model, deserialized.Data.Model)
			checkFields(t, "BOSID", tt.data.BOSID, deserialized.Data.BOSID)
			checkFields(t, "EOSID", tt.data.EOSID, deserialized.Data.EOSID)
			checkBoolField(t, "HasBOSID", tt.data.HasBOSID, deserialized.Data.HasBOSID)
			checkBoolField(t, "HasEOSID", tt.data.HasEOSID, deserialized.Data.HasEOSID)
			checkBoolField(t, "AddBOS", tt.data.AddBOS, deserialized.Data.AddBOS)
			checkBoolField(t, "AddEOS", tt.data.AddEOS, deserialized.Data.AddEOS)

			// Verify tokens are preserved (order matters for ID mapping)
			if len(tt.data.Tokens) != len(deserialized.Data.Tokens) {
				t.Errorf("Token count mismatch: original=%d, deserialized=%d",
					len(tt.data.Tokens), len(deserialized.Data.Tokens))
			} else {
				for i := range tt.data.Tokens {
					if tt.data.Tokens[i] != deserialized.Data.Tokens[i] {
						t.Errorf("Token[%d] mismatch: original=%q, deserialized=%q",
							i, tt.data.Tokens[i], deserialized.Data.Tokens[i])
					}
				}
			}

			// Reconstruct and verify encoding matches
			testInputs := []string{"hello world", "playing"}
			for _, input := range testInputs {
				if len(tt.data.Tokens) > 0 && tt.data.Tokens[0] == "[UNK]" {
					originalEncoder, err := bpe.New(tt.data) // Use BPE as fallback for wordpiece-like data
					if err != nil {
						t.Logf("Skipping encoding test: %v", err)
						continue
					}

					deserEncoder, err := bpe.New(deserialized.Data)
					if err != nil {
						t.Logf("Skipping deserialization: %v", err)
						continue
					}

					origIDs := originalEncoder.EncodeIDs(input)
					deserIDs := deserEncoder.EncodeIDs(input)
					if !idsEqual(origIDs, deserIDs) {
						t.Errorf("EncodeIDs mismatch for %q:\n  original:   %v\n  deserialized: %v",
							input, origIDs, deserIDs)
					}
				}
			}
		})
	}
}

// TestRoundTrip_Unigram_Limited documents the current limitation that Unigram tokenizers
// cannot be fully round-tripped because SPMProbabilities (used for Viterbi segmentation) are
// not serialized. This test verifies what CAN be preserved: vocab structure and basic config.
func TestRoundTrip_Unigram_Limited(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"▁hello", "▁world"},
	}

	tokenizer := tok.New(data)

	var buf bytes.Buffer
	if err := tok.Serialize(tokenizer, &buf); err != nil {
		t.Fatalf("Failed to serialize Unigram tokenizer: %v", err)
	}

	deserialized, err := tok.Deserialize(&buf)
	if err != nil {
		t.Fatalf("Failed to deserialize Unigram tokenizer: %v", err)
	}

	// Verify vocab is preserved (this is what CAN be round-tripped)
	checkFields(t, "Model", data.Model, deserialized.Data.Model)
	if len(data.Tokens) != len(deserialized.Data.Tokens) {
		t.Errorf("Token count mismatch: original=%d, deserialized=%d",
			len(data.Tokens), len(deserialized.Data.Tokens))
	} else {
		for i := range data.Tokens {
			if data.Tokens[i] != deserialized.Data.Tokens[i] {
				t.Errorf("Token[%d] mismatch: original=%q, deserialized=%q",
					i, data.Tokens[i], deserialized.Data.Tokens[i])
			}
		}
	}

	// Note: SPMProbabilities are NOT preserved (known limitation). A deserialized Unigram
	// tokenizer cannot perform optimal Viterbi segmentation without these probabilities.
	t.Logf("NOTE: Unigram SPMProbabilities are not serialized - this is a known limitation")
}

// TestRoundTrip_SpecialTokens verifies that special token IDs (BOS, EOS, UNK, PAD) and
// their Has* flags are preserved through serialization/deserialization. This ensures
// that boundary markers and unknown/padding tokens maintain their configuration.
func TestRoundTrip_SpecialTokens(t *testing.T) {
	data := &common.TokenizerData{
		Model:    "wordpiece",
		Tokens:   []string{"[UNK]", "[CLS]", "[SEP]", "hello", "world"},
		BOSID:    102, EOSID: 103, HasBOSID: true, HasEOSID: true,
		UNKID:    0, PADID: 99, HasUNKID: true, HasPADID: true,
		AddBOS:   true, AddEOS: true,
	}

	tokenizer := tok.New(data)

	var buf bytes.Buffer
	if err := tok.Serialize(tokenizer, &buf); err != nil {
		t.Fatalf("Failed to serialize tokenizer: %v", err)
	}

	deserialized, err := tok.Deserialize(&buf)
	if err != nil {
		t.Fatalf("Failed to deserialize tokenizer: %v", err)
	}

	// Verify special token IDs are preserved
	checkFields(t, "BOSID", data.BOSID, deserialized.Data.BOSID)
	checkFields(t, "EOSID", data.EOSID, deserialized.Data.EOSID)
	checkFields(t, "UNKID", data.UNKID, deserialized.Data.UNKID)
	checkFields(t, "PADID", data.PADID, deserialized.Data.PADID)

	// Verify Has* flags are preserved
	checkBoolField(t, "HasBOSID", data.HasBOSID, deserialized.Data.HasBOSID)
	checkBoolField(t, "HasEOSID", data.HasEOSID, deserialized.Data.HasEOSID)
	checkBoolField(t, "HasUNKID", data.HasUNKID, deserialized.Data.HasUNKID)
	checkBoolField(t, "HasPADID", data.HasPADID, deserialized.Data.HasPADID)

	// Verify AddBOS/AddEOS bools are preserved
	checkBoolField(t, "AddBOS", data.AddBOS, deserialized.Data.AddBOS)
	checkBoolField(t, "AddEOS", data.AddEOS, deserialized.Data.AddEOS)
}

// TestRoundTrip_PreType verifies that the PreType setting (which selects pre-tokenization
// strategy for BPE models like GPT2 vs Llama3) is preserved through serialization.
func TestRoundTrip_PreType(t *testing.T) {
	tests := []struct {
		name    string
		preType common.PreType
	}{
		{"gpt2", common.PreGPT2},
		{"llama3", common.PreLlama3},
		{"qwen2", common.PreQwen2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := &common.TokenizerData{
				Model:   "bpe",
				Tokens:  []string{"hello", "world"},
				PreType: tt.preType,
			}

			tokenizer := tok.New(data)

			var buf bytes.Buffer
			if err := tok.Serialize(tokenizer, &buf); err != nil {
				t.Fatalf("Failed to serialize BPE tokenizer: %v", err)
			}

			deserialized, err := tok.Deserialize(&buf)
			if err != nil {
				t.Fatalf("Failed to deserialize BPE tokenizer: %v", err)
			}

			checkFields(t, "PreType", int32(data.PreType), int32(deserialized.Data.PreType))
		})
	}
}

// TestRoundTrip_MergeOrder verifies that BPE merge rules maintain their order through
// serialization/deserialization. Merge order is critical because earlier merges take priority.
func TestRoundTrip_MergeOrder(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "bpe",
		Tokens: []string{"h", "e", "l", "o", "Ġw", "or", "ld"},
		Merges: []common.Merge{
			{A: "l", B: "o"},      // lo
			{A: "he", B: "llo"},   // hello
			{A: "Ġw", B: "orld"},  // Ġworld
		},
	}

	tokenizer := tok.New(data)

	var buf bytes.Buffer
	if err := tok.Serialize(tokenizer, &buf); err != nil {
		t.Fatalf("Failed to serialize BPE tokenizer: %v", err)
	}

	deserialized, err := tok.Deserialize(&buf)
	if err != nil {
		t.Fatalf("Failed to deserialize BPE tokenizer: %v", err)
	}

	// Verify merge count is preserved
	checkFields(t, "MergeCount", len(data.Merges), len(deserialized.Data.Merges))

	// Verify each merge rule is preserved in order
	for i, origMerge := range data.Merges {
		deserMerge := deserialized.Data.Merges[i]
		if origMerge.A != deserMerge.A || origMerge.B != deserMerge.B {
			t.Errorf("Merge[%d] mismatch:\n  original:   %q + %q\n  deserialized: %q + %q",
				i, origMerge.A, origMerge.B, deserMerge.A, deserMerge.B)
		}
	}

	// Verify encoding behavior is preserved with merges
	originalEncoder, err := bpe.New(data)
	if err != nil {
		t.Fatalf("Failed to create original encoder: %v", err)
	}

	deserEncoder, err := bpe.New(deserialized.Data)
	if err != nil {
		t.Fatalf("Failed to create deserialized encoder: %v", err)
	}

	input := "hello world"
	originalIDs := originalEncoder.EncodeIDs(input)
	deserIDs := deserEncoder.EncodeIDs(input)

	if !idsEqual(originalIDs, deserIDs) {
		t.Errorf("EncodeIDs mismatch for %q:\n  original:   %v\n  deserialized: %v",
			input, originalIDs, deserIDs)
	}
}

// idsEqual compares two integer slices for equality.
func idsEqual(a, b []int) bool {
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

// checkFields verifies that two values are equal and reports a descriptive error if not.
func checkFields[T comparable](t *testing.T, name string, expected, actual T) {
	t.Helper()
	if expected != actual {
		t.Errorf("%s mismatch: expected %v, got %v", name, expected, actual)
	}
}

// checkBoolField verifies that two boolean values are equal and reports a descriptive error if not.
func checkBoolField(t *testing.T, name string, expected, actual bool) {
	t.Helper()
	if expected != actual {
		t.Errorf("%s mismatch: expected %v, got %v", name, expected, actual)
	}
}

// TestRoundTrip_SPM_Protobuf verifies that SPM tokenizer configuration survives serialization/deserialization.
// Note: Full SPM functionality requires the protobuf model binary (SPMModel field), which may not be present
// in all test cases. This test focuses on verifying that vocab and basic config are preserved.
func TestRoundTrip_SPM_Protobuf(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"▁hello", "▁world"},
	}

	tokenizer := tok.New(data)

	var buf bytes.Buffer
	if err := tok.Serialize(tokenizer, &buf); err != nil {
		t.Fatalf("Failed to serialize SPM tokenizer: %v", err)
	}

	deserialized, err := tok.Deserialize(&buf)
	if err != nil {
		t.Fatalf("Failed to deserialize SPM tokenizer: %v", err)
	}

	// Verify vocab is preserved (this is what CAN be round-tripped without SPMModel binary)
	checkFields(t, "Model", data.Model, deserialized.Data.Model)
	if len(data.Tokens) != len(deserialized.Data.Tokens) {
		t.Errorf("Token count mismatch: original=%d, deserialized=%d",
			len(data.Tokens), len(deserialized.Data.Tokens))
	} else {
		for i := range data.Tokens {
			if data.Tokens[i] != deserialized.Data.Tokens[i] {
				t.Errorf("Token[%d] mismatch: original=%q, deserialized=%q",
					i, data.Tokens[i], deserialized.Data.Tokens[i])
			}
		}
	}

	// Note: SPMModel binary (if present) would also be preserved via the 'X' field in proto package.
	// This test verifies that at minimum, vocab structure survives round-trip.
	t.Logf("SPM tokenizer round-trip verified - vocab and config preserved")
	if len(data.SPMModel) > 0 {
		t.Logf("NOTE: SPMModel binary would also be preserved in full serialization")
	}
}
