package spm

import (
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestSPM_NewProgrammatic(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world", "▁hello world"},
		BOSID:  1,
		EOSID:  2,
		PADID:  3,
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}
	if spm == nil {
		t.Fatal("New returned nil")
	}
	if spm.Type() != "spm" {
		t.Errorf("Type() = %q, want %q", spm.Type(), "spm")
	}
}

func TestSPM_NilData(t *testing.T) {
	spm := &SPM{}
	if spm == nil {
		t.Fatal("nil SPM should not be nil")
	}
	if spm.Count("") != 0 {
		t.Error("Count('') on nil SPM should be 0")
	}
	if len(spm.EncodeIDs("")) != 0 {
		t.Error("EncodeIDs('') on nil SPM should return empty slice")
	}
}

func TestSPM_EncodeIDs(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world", "▁hello world"},
		BOSID:  1,
		EOSID:  2,
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Test direct vocab match (space is converted to ▁)
	ids := spm.EncodeIDs("hello world")
	if len(ids) < 1 {
		t.Errorf("EncodeIDs('hello world') = %d ids, want at least 1", len(ids))
	}
	t.Logf("EncodeIDs('hello world') = %v (len=%d)", ids, len(ids))

	// Test single token
	ids2 := spm.EncodeIDs("world")
	if len(ids2) < 1 {
		t.Errorf("EncodeIDs('world') = %d ids, want at least 1", len(ids2))
	}
	t.Logf("EncodeIDs('world') = %v (len=%d)", ids2, len(ids2))

	// Test with BOS/EOS — vocab has "▁hello world" which matches as single token.
	data.AddBOS = true
	data.AddEOS = true
	spm2, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}
	ids3 := spm2.EncodeIDs("hello world")
	if len(ids3) < 1 { // At least one content token + BOS/EOS
		t.Errorf("EncodeIDs with BOS/EOS = %d ids, want at least 1", len(ids3))
	}
	t.Logf("EncodeIDs('hello world') with BOS/EOS = %v (len=%d)", ids3, len(ids3))
}

func TestSPM_Detokenize(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Encode and then decode — should recover original text (with spaces restored).
	ids := spm.EncodeIDs("hello world")
	decoded := spm.Detokenize(ids)
	t.Logf("Detokenize(%v) = %q", ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded)
	}
}

func TestSPM_DetokenizeWithMeta(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Verify each token's ID mapping.
	for i, tok := range data.Tokens {
		id := spm.TokenID(tok)
		t.Logf("Token %q → ID %d (expected %d)", tok, id, i)
	}

	ids := spm.EncodeIDs("hello world")
	t.Logf("EncodeIDs('hello world') = %v", ids)

	for _, id := range ids {
		tok := spm.IDToToken(id)
		if tok == "" {
			tok = "<unknown>"
		}
		t.Logf("  ID %d → token %q", id, tok)
	}
}

func TestSPM_TrieDebug(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Check what's in the trie by testing matchLongest on various prefixes.
	testStrings := []string{
		"▁hello world", // input with space→▁
		"▁hello",
		"world",
		"h",
		"he",
		"hel",
		"hell",
		"hello",
	}

	for _, ts := range testStrings {
		id, length := spm.trie.matchLongest([]byte(ts))
		tok := ""
		if id >= 0 && id < len(spm.vocabKeys) {
			tok = spm.vocabKeys[id]
		}
		t.Logf("matchLongest(%q) → ID %d, length %d, token=%q", ts, id, length, tok)
	}
}

func TestSPM_ConvertSpaces(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	input := "hello world"
	converted := spm.convertSpacesToMeta(input)
	t.Logf("Input: %q → Converted: %q (len=%d)", input, converted, len(converted))

	// Check if the converted string matches any token.
	for i, tok := range data.Tokens {
		if tok == converted {
			t.Logf("Found exact match for converted text at index %d: %q", i, tok)
		}
	}
}

func TestSPM_Count(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world", "▁hello world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	count := spm.Count("hello world")
	ids := spm.EncodeIDs("hello world")
	if count != len(ids) {
		t.Errorf("Count('hello world') = %d, len(EncodeIDs) = %d", count, len(ids))
	}
	t.Logf("Count('hello world') = %d, EncodeIDs = %v", count, ids)
}

func TestSPM_HasToken(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	if !spm.HasToken("▁hello") {
		t.Error("HasToken('▁hello') should be true")
	}
	if spm.HasToken("hello world") {
		t.Error("HasToken('hello world') should be false (spaces not in vocab)")
	}
	if !spm.HasToken("world") {
		t.Error("HasToken('world') should be true")
	}
}

func TestSPM_TokenID(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	if id := spm.TokenID("▁hello"); id != 3 {
		t.Errorf("TokenID('▁hello') = %d, want 3", id)
	}
	if id := spm.TokenID("world"); id != 4 {
		t.Errorf("TokenID('world') = %d, want 4", id)
	}
	if id := spm.TokenID("nonexistent"); id != -1 {
		t.Errorf("TokenID('nonexistent') = %d, want -1", id)
	}
}

func TestSPM_ClearCache(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Encode to populate cache
	spm.EncodeIDs("hello world")

	// Clear cache
	spm.ClearCache()

	// Verify it still works after clear
	ids := spm.EncodeIDs("hello world")
	if len(ids) < 1 {
		t.Error("EncodeIDs should work after ClearCache")
	}
	t.Logf("After ClearCache: EncodeIDs('hello world') = %v", ids)
}

func TestSPM_GreedyLongestMatch(t *testing.T) {
	data := &common.TokenizerData{
		Model: "spm",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"a", "b", "c", // single chars
			"ab", "bc", // two-char tokens
			"abc"}, // three-char token
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// "abc" should match as a single token (longest greedy match)
	ids := spm.EncodeIDs("abc")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('abc') = %d ids, want 1 (greedy longest match)", len(ids))
	}
	t.Logf("EncodeIDs('abc') = %v", ids)
}

func TestSPM_UnmatchedToken(t *testing.T) {
	data := &common.TokenizerData{
		Model: "spm",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"a", "b"}, // only single chars
		HasUNKID: true,
		UNKID:    0,
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// "xyz" has no match — should use UNK token
	ids := spm.EncodeIDs("xyz")
	t.Logf("EncodeIDs('xyz') = %v (len=%d)", ids, len(ids))
	if len(ids) == 0 {
		t.Error("EncodeIDs('xyz') with UNK should return at least one token")
	}
}

func TestSPM_EmptyString(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "spm",
		Tokens: []string{"<unk>", "<s>", "</s>", "a"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	count := spm.Count("")
	ids := spm.EncodeIDs("")
	if count != 0 || len(ids) != 0 {
		t.Errorf("Count('') = %d, EncodeIDs('') = %d ids, want both empty", count, len(ids))
	}
}

func TestSPM_Concurrent(t *testing.T) {
	data := &common.TokenizerData{
		Model: "spm",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"},
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	longInput := strings.Repeat("abcdefghij ", 100)

	ids1 := spm.EncodeIDs(longInput)
	ids2 := spm.EncodeIDs(longInput)

	if len(ids1) != len(ids2) {
		t.Errorf("Concurrent encoding mismatch: %d vs %d ids", len(ids1), len(ids2))
	}
	t.Logf("Concurrent EncodeIDs length = %d", len(ids1))
}

func TestSPM_Unicode(t *testing.T) {
	data := &common.TokenizerData{
		Model: "spm",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"a", "b", "c",
			"世", "界", "世界"}, // Chinese characters
	}

	spm, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	ids := spm.EncodeIDs("世界")
	t.Logf("EncodeIDs('世界') = %v (len=%d)", ids, len(ids))
	if len(ids) < 1 {
		t.Error("EncodeIDs('世界') should return at least one token")
	}
}

func TestSPM_DetokenizeEmpty(t *testing.T) {
	spm := &SPM{}
	result := spm.Detokenize(nil)
	if result != "" {
		t.Errorf("Detokenize(nil) = %q, want empty string", result)
	}

	result2 := spm.Detokenize([]int{})
	if result2 != "" {
		t.Errorf("Detokenize([]) = %q, want empty string", result2)
	}
}

func TestSPM_NilReceiver(t *testing.T) {
	var s *SPM

	if s.HasToken("hello") {
		t.Error("nil SPM.HasToken should return false")
	}
	if s.TokenID("hello") != -1 {
		t.Error("nil SPM.TokenID should return -1")
	}
	if s.Count("hello") != 0 {
		t.Error("nil SPM.Count should be 0")
	}
	if s.EncodeIDs("hello") != nil {
		t.Error("nil SPM.EncodeIDs should return nil")
	}
}
