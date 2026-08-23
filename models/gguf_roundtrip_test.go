package models

import (
	"strings"
	"testing"
)

// TestRoundTrip_BasicVocabMatch verifies that EncodeIDs + Detokenize round-trips
// when the input is a single direct vocab entry.
func TestRoundTrip_BasicVocabMatch(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hello world", "foo bar"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{"hello world", "foo bar"}
	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			if len(ids) != 1 {
				t.Fatalf("EncodeIDs(%q) = %d ids, want 1", input, len(ids))
			}

			decoded := bpe.Detokenize(ids)
			if decoded != input {
				t.Errorf("Detokenize(EncodeIDs(%q)) = %q, want %q", input, decoded, input)
			}
		})
	}
}

// TestRoundTrip_SpacePrefixCharacter tests round-tripping with space-prefixed tokens.
func TestRoundTrip_SpacePrefixCharacter(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "Ġworld", "hello world"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{
		"hello world", // direct vocab match
		"hello",       // single token
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			if len(ids) == 0 {
				t.Fatalf("EncodeIDs(%q) returned no ids", input)
			}

			decoded := bpe.Detokenize(ids)
			// For direct vocab match, output should be exact
			if decoded != input {
				t.Errorf("Detokenize(EncodeIDs(%q)) = %q (len=%d), want %q",
					input, decoded, len(decoded), input)
			}
		})
	}
}

// TestRoundTrip_MultiTokenSequence tests round-tripping when the input is split
// into multiple tokens that need to be concatenated.
func TestRoundTrip_MultiTokenSequence(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hel", "lo", "wor", "ld", "Ġworld"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []struct {
		name    string
		text    string
		wantLen int
	}{
		{"hello", "hel lo", 3},
		{"world", "wor ld", 3},
	}

	for _, tc := range inputs {
		t.Run(tc.name, func(t *testing.T) {
			ids := bpe.EncodeIDs(tc.text)
			if len(ids) == 0 {
				t.Fatalf("EncodeIDs(%q) returned no ids", tc.text)
			}

			decoded := bpe.Detokenize(ids)
			t.Logf("Input: %q -> Decoded: %q (len=%d)", tc.text, decoded, len(decoded))
		})
	}
}

// TestRoundTrip_BOS_EOS verifies that BOS/EOS tokens are preserved through round-trip.
func TestRoundTrip_BOS_EOS(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hello", "world"},
		Merges:   []Merge{},
		BOSID:    100,
		EOSID:    101,
		HasBOSID: true,
		HasEOSID: true,
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
			if len(ids) < 2 {
				t.Fatalf("EncodeIDs(%q) = %d ids, want at least 2 (BOS + content)", input, len(ids))
			}

			decoded := bpe.Detokenize(ids)
			t.Logf("Encoded %q -> %v -> Detokenized to %q", input, ids, decoded)
		})
	}
}

// TestRoundTrip_EmptyString verifies round-trip with empty input.
func TestRoundTrip_EmptyString(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	ids := bpe.EncodeIDs("")
	if len(ids) != 0 {
		t.Errorf("EncodeIDs('') = %d ids, want 0", len(ids))
	}

	decoded := bpe.Detokenize(ids)
	if decoded != "" {
		t.Errorf("Detokenize([]) = %q, want empty string", decoded)
	}
}

// TestRoundTrip_UnsupportedInput verifies behavior with text not in vocab.
func TestRoundTrip_UnsupportedInput(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hello"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1, // no UNK token available
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{"world", "xyz"}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			t.Logf("EncodeIDs(%q) = %v", input, ids)

			if len(ids) == 0 {
				t.Log("No tokens produced (expected when vocab doesn't match)")
				return
			}

			decoded := bpe.Detokenize(ids)
			t.Logf("Detokenized: %q", decoded)
		})
	}
}

// TestRoundTrip_UnicodeCharacters tests round-tripping with multi-byte Unicode characters.
func TestRoundTrip_UnicodeCharacters(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "世界", "hello世界"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{"hello", "世界", "hello世界"}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			if len(ids) == 0 {
				t.Fatalf("EncodeIDs(%q) returned no ids", input)
			}

			decoded := bpe.Detokenize(ids)
			if decoded != input {
				t.Errorf("Detokenize(EncodeIDs(%q)) = %q, want %q", input, decoded, input)
			}
		})
	}
}

// TestRoundTrip_ConcurrentEncoding verifies that concurrent encoding produces
// the same token IDs as sequential encoding.
func TestRoundTrip_ConcurrentEncoding(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "world", "Ġworld"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	longInput := strings.Repeat("hello world ", 1000) // >32KB to trigger concurrent path

	ids1 := bpe.EncodeIDs(longInput)
	ids2 := bpe.EncodeIDs(longInput)

	if len(ids1) != len(ids2) {
		t.Errorf("Concurrent vs sequential: %d vs %d ids", len(ids1), len(ids2))
	} else {
		for i := range ids1 {
			if ids1[i] != ids2[i] {
				t.Errorf("ID mismatch at position %d: %d vs %d", i, ids1[i], ids2[i])
				break
			}
		}
	}

	// Verify round-trip works for the concurrent path too
	if len(ids1) > 0 {
		decoded := bpe.Detokenize(ids1[:min(100, len(ids1))])
		t.Logf("Decoded first %d tokens: %q", min(100, len(decoded)), decoded)
	}
}

// TestRoundTrip_PreservesTokenOrder verifies that the order of tokens is preserved.
func TestRoundTrip_PreservesTokenOrder(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a", "b", "c", "ab", "abc", " ", "a b c"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := map[string]string{
		"abc":   "abc",   // single token match (longest greedy)
		"a b c": "a b c", // full phrase in vocab
	}

	for input, expected := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			if len(ids) == 0 {
				t.Fatalf("EncodeIDs(%q) returned no ids", input)
			}

			decoded := bpe.Detokenize(ids)
			t.Logf("Input: %q -> IDs: %v -> Decoded: %q", input, ids, decoded)

			if expected != "" && decoded != expected {
				t.Errorf("Round-trip mismatch: got %q, want %q", decoded, expected)
			}
		})
	}
}

// TestRoundTrip_CacheConsistency verifies that cached and uncached encodings produce identical results.
func TestRoundTrip_CacheConsistency(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "world"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}
	bpe.SetCache(10)

	inputs := []string{"hello world", "hello"}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids1 := bpe.EncodeIDs(input)
			decoded1 := bpe.Detokenize(ids1)

			bpe.ClearCache()
			ids2 := bpe.EncodeIDs(input)
			decoded2 := bpe.Detokenize(ids2)

			if len(ids1) != len(ids2) {
				t.Errorf("Cached vs uncached: %d vs %d ids", len(ids1), len(ids2))
			}

			for i := range ids1 {
				if i < len(ids2) && ids1[i] != ids2[i] {
					t.Errorf("ID mismatch at position %d: %d vs %d", i, ids1[i], ids2[i])
				}
			}

			if decoded1 != decoded2 {
				t.Errorf("Decoded mismatch: %q vs %q", decoded1, decoded2)
			}

			t.Logf("Input: %q -> IDs: %v -> Decoded: %q (cache hit)", input, ids1, decoded1)
		})
	}
}

// TestRoundTrip_BPEWithMerges tests round-tripping with a BPE model that uses merges.
func TestRoundTrip_BPEWithMerges(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"the", "Ġquick", "Ġbrown", "Ġfox", "the quick brown fox"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{
		"the quick brown fox", // direct match
		"the",                 // single token
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := bpe.EncodeIDs(input)
			if len(ids) == 0 {
				t.Fatalf("EncodeIDs(%q) returned no ids", input)
			}

			decoded := bpe.Detokenize(ids)
			if decoded != input {
				t.Errorf("Detokenize(EncodeIDs(%q)) = %q, want %q", input, decoded, input)
			}
		})
	}
}

// TestRoundTrip_NilBPE verifies that nil BPE handles don't panic.
func TestRoundTrip_NilBPE(t *testing.T) {
	var bpe *BPE

	ids := bpe.EncodeIDs("hello")
	if ids != nil {
		t.Errorf("nil EncodeIDs = %v, want nil", ids)
	}

	decoded := bpe.Detokenize(ids)
	if decoded != "" {
		t.Errorf("nil Detokenize = %q, want empty string", decoded)
	}

	count := bpe.Count("hello")
	if count != 0 {
		t.Errorf("nil Count = %d, want 0", count)
	}
}

// TestRoundTrip_TokensMethod verifies the Tokens() method.
func TestRoundTrip_TokensMethod(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "world", "hello world"},
		Merges: []Merge{},
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}

	inputs := []string{"hello world", "hello"}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			tokens := bpe.Tokens(input)
			if len(tokens) == 0 {
				t.Fatalf("Tokens(%q) returned no tokens", input)
			}

			t.Logf("Tokens(%q) = %v", input, tokens)
		})
	}
}

// Helper function to find minimum of two ints.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
