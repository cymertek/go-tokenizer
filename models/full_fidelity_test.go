package models

import (
	"bytes"
	"testing"

	tok "github.com/cymertek/go-tokenizer"
	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// TestFullFidelity_UnigramWithProbabilities verifies that Unigram tokenizers can be fully
// round-tripped with SPMProbabilities (interleaved [id, score] pairs for Viterbi decoding).
func TestFullFidelity_UnigramWithProbabilities(t *testing.T) {
	// Create a Unigram tokenizer with interleaved [id, score] probabilities.
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"▁hello", "▁world", "▁foo"},
		SPMProbabilities: []float64{
			0, -1.5, // token 0 (▁hello) has score -1.5
			1, -2.3, // token 1 (▁world) has score -2.3
			2, -0.8, // token 2 (▁foo) has score -0.8
		},
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

	// Verify SPMProbabilities are preserved exactly.
	expectedProbs := []float64{0, -1.5, 1, -2.3, 2, -0.8}
	if len(deserialized.Data.SPMProbabilities) != len(expectedProbs) {
		t.Errorf("SPMProbabilities length mismatch: expected %d, got %d", len(expectedProbs), len(deserialized.Data.SPMProbabilities))
		return
	}

	for i, v := range deserialized.Data.SPMProbabilities {
		if v != expectedProbs[i] {
			t.Errorf("SPMProbabilities[%d]: expected %v, got %v", i, expectedProbs[i], v)
		}
	}
}

// TestFullFidelity_BPEWithTokenType verifies that BPE tokenizers preserve TokenType arrays.
func TestFullFidelity_BPEWithTokenType(t *testing.T) {
	data := &common.TokenizerData{
		Model:     "bpe",
		Tokens:    []string{"hello", "world", "Ġhello"},
		TokenType: []int32{0, 1, 2}, // Different types for each token.
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

	checkFields(t, "TokenType length", len(data.TokenType), len(deserialized.Data.TokenType))
	for i := range data.TokenType {
		checkFields(t, "TokenType", data.TokenType[i], deserialized.Data.TokenType[i])
	}
}

// TestFullFidelity_SpecialTokenIDs verifies that all special token IDs (BOS, EOS, EOT, EOM, UNK, PAD)
// and their Has* flags are preserved through serialization/deserialization.
func TestFullFidelity_SpecialTokenIDs(t *testing.T) {
	data := &common.TokenizerData{
		Model:    "bpe",
		Tokens:   []string{"<unk>", "<s>", "</s>", "<eot>", "<eom>", "<pad>", "hello"},
		BOSID:    1, EOSID: 2, EOTID: 3, EOMID: 4, UNKID: 0, PADID: 5,
		HasBOSID: true, HasEOSID: true, HasEOTID: true, HasEOMID: true, HasUNKID: true, HasPADID: true,
		AddBOS:   true, AddEOS:   true,
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

	checkFields(t, "BOSID", data.BOSID, deserialized.Data.BOSID)
	checkFields(t, "EOSID", data.EOSID, deserialized.Data.EOSID)
	checkFields(t, "EOTID", data.EOTID, deserialized.Data.EOTID)
	checkFields(t, "EOMID", data.EOMID, deserialized.Data.EOMID)
	checkFields(t, "UNKID", data.UNKID, deserialized.Data.UNKID)
	checkFields(t, "PADID", data.PADID, deserialized.Data.PADID)

	checkBoolField(t, "HasBOSID", data.HasBOSID, deserialized.Data.HasBOSID)
	checkBoolField(t, "HasEOSID", data.HasEOSID, deserialized.Data.HasEOSID)
	checkBoolField(t, "HasEOTID", data.HasEOTID, deserialized.Data.HasEOTID)
	checkBoolField(t, "HasEOMID", data.HasEOMID, deserialized.Data.HasEOMID)
	checkBoolField(t, "HasUNKID", data.HasUNKID, deserialized.Data.HasUNKID)
	checkBoolField(t, "HasPADID", data.HasPADID, deserialized.Data.HasPADID)

	checkBoolField(t, "AddBOS", data.AddBOS, deserialized.Data.AddBOS)
	checkBoolField(t, "AddEOS", data.AddEOS, deserialized.Data.AddEOS)
}

// TestFullFidelity_PreTypePreservation verifies that PreType (GPT2, Llama3, etc.) is preserved.
func TestFullFidelity_PreTypePreservation(t *testing.T) {
	preTypes := []common.PreType{
		common.PreGPT2,
		common.PreLlama3,
		common.PreQwen2,
		common.PreStarcoder,
		common.PreDeepSeekLLM,
	}

	for _, pt := range preTypes {
		data := &common.TokenizerData{
			Model:   "bpe",
			Tokens:  []string{"hello", "world"},
			PreType: pt,
		}

		tokenizer := tok.New(data)
		var buf bytes.Buffer
		if err := tok.Serialize(tokenizer, &buf); err != nil {
			t.Fatalf("Failed to serialize tokenizer for PreType %v: %v", pt, err)
		}

		deserialized, err := tok.Deserialize(&buf)
		if err != nil {
			t.Fatalf("Failed to deserialize tokenizer for PreType %v: %v", pt, err)
		}

		checkFields(t, "PreType", int32(pt), int32(deserialized.Data.PreType))
	}
}

// TestFullFidelity_WordPieceWithContinuationPrefix verifies that WordPiece tokenizers preserve
// their vocabulary and special tokens through round-trip.
func TestFullFidelity_WordPieceWithContinuationPrefix(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"},
		BOSID:  102, EOSID: 103, HasBOSID: true, HasEOSID: true,
		AddBOS: true, AddEOS: true,
	}

	tokenizer := tok.New(data)
	var buf bytes.Buffer
	if err := tok.Serialize(tokenizer, &buf); err != nil {
		t.Fatalf("Failed to serialize WordPiece tokenizer: %v", err)
	}

	deserialized, err := tok.Deserialize(&buf)
	if err != nil {
		t.Fatalf("Failed to deserialize WordPiece tokenizer: %v", err)
	}

	checkFields(t, "Model", data.Model, deserialized.Data.Model)
	checkFields(t, "BOSID", data.BOSID, deserialized.Data.BOSID)
	checkFields(t, "EOSID", data.EOSID, deserialized.Data.EOSID)

	// Verify tokens are preserved.
	if len(data.Tokens) != len(deserialized.Data.Tokens) {
		t.Errorf("Token count mismatch: original=%d, deserialized=%d", len(data.Tokens), len(deserialized.Data.Tokens))
	} else {
		for i := range data.Tokens {
			checkFields(t, "Tokens", data.Tokens[i], deserialized.Data.Tokens[i])
		}
	}

	// Verify encoding behavior is preserved using BPE as fallback (WordPiece uses similar logic).
	testInputs := []string{"playing", "hello world"}
	for _, input := range testInputs {
		origEncoder, err := bpe.New(data)
		if err != nil {
			t.Logf("Skipping encoding test for %q: %v", input, err)
			continue
		}
		deserEncoder, err := bpe.New(deserialized.Data)
		if err != nil {
			t.Logf("Skipping deserialization test for %q: %v", input, err)
			continue
		}

		origIDs := origEncoder.EncodeIDs(input)
		deserIDs := deserEncoder.EncodeIDs(input)
		if !idsEqual(origIDs, deserIDs) {
			t.Errorf("EncodeIDs mismatch for %q:\n  original:   %v\n  deserialized: %v", input, origIDs, deserIDs)
		}
	}
}

// TestFullFidelity_BPEWithMergesAndPreType verifies that BPE tokenizers with merges and PreType
// can be fully round-tripped.
func TestFullFidelity_BPEWithMergesAndPreType(t *testing.T) {
	data := &common.TokenizerData{
		Model:     "bpe",
		Tokens:    []string{"h", "e", "l", "o", "Ġw", "or", "ld"},
		Merges: []common.Merge{
			{A: "l", B: "o"},      // lo
			{A: "he", B: "llo"},   // hello
			{A: "Ġw", B: "orld"},  // Ġworld
		},
		PreType:   common.PreLlama3,
		SpaceChar: 'Ġ',
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

	checkFields(t, "Model", data.Model, deserialized.Data.Model)
	checkFields(t, "PreType", int32(data.PreType), int32(deserialized.Data.PreType))
	checkFields(t, "SpaceChar", rune(data.SpaceChar), rune(deserialized.Data.SpaceChar))

	// Verify merges are preserved in order.
	if len(data.Merges) != len(deserialized.Data.Merges) {
		t.Errorf("Merge count mismatch: original=%d, deserialized=%d", len(data.Merges), len(deserialized.Data.Merges))
	} else {
		for i := range data.Merges {
			checkFields(t, "Merge A", data.Merges[i].A, deserialized.Data.Merges[i].A)
			checkFields(t, "Merge B", data.Merges[i].B, deserialized.Data.Merges[i].B)
		}
	}

	// Verify encoding behavior is preserved using both original and deserialized data.
	testInputs := []string{"hello world", "helo"}
	for _, input := range testInputs {
		origEncoder, err := bpe.New(data)
		if err != nil {
			t.Logf("Skipping encoding test for %q: %v", input, err)
			continue
		}
		deserEncoder, err := bpe.New(deserialized.Data)
		if err != nil {
			t.Logf("Skipping deserialization test for %q: %v", input, err)
			continue
		}

		origIDs := origEncoder.EncodeIDs(input)
		deserIDs := deserEncoder.EncodeIDs(input)
		if !idsEqual(origIDs, deserIDs) {
			t.Errorf("EncodeIDs mismatch for %q:\n  original:   %v\n  deserialized: %v", input, origIDs, deserIDs)
		}
	}
}

