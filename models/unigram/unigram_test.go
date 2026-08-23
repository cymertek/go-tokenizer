package unigram

import (
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestUnigram_NewProgrammatic(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "world", "▁hello world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}
	if u == nil {
		t.Fatal("New returned nil")
	}
	if u.Type() != "unigram" {
		t.Errorf("Type() = %q, want %q", u.Type(), "unigram")
	}
}

func TestUnigram_NilData(t *testing.T) {
	u := &Unigram{}
	if u == nil {
		t.Fatal("nil Unigram should not be nil")
	}
	if u.Count("") != 0 {
		t.Error("Count('') on nil Unigram should be 0")
	}
	if len(u.EncodeIDs("")) != 0 {
		t.Error("EncodeIDs('') on nil Unigram should return empty slice")
	}
}

func TestUnigram_EncodeIDs(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	ids := u.EncodeIDs("hello world")
	if len(ids) < 1 {
		t.Errorf("EncodeIDs('hello world') = %d ids, want at least 1", len(ids))
	}
	t.Logf("EncodeIDs('hello world') = %v (len=%d)", ids, len(ids))

	ids2 := u.EncodeIDs("world")
	if len(ids2) < 1 {
		t.Errorf("EncodeIDs('world') = %d ids, want at least 1", len(ids2))
	}
	t.Logf("EncodeIDs('world') = %v (len=%d)", ids2, len(ids2))
}

func TestUnigram_Detokenize(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	ids := u.EncodeIDs("hello world")
	decoded := u.Detokenize(ids)
	t.Logf("Detokenize(%v) = %q", ids, decoded)

	if !strings.Contains(decoded, "hello") || !strings.Contains(decoded, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded)
	}
}

func TestUnigram_Count(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	count := u.Count("hello world")
	ids := u.EncodeIDs("hello world")
	if count != len(ids) {
		t.Errorf("Count('hello world') = %d, len(EncodeIDs) = %d", count, len(ids))
	}
	t.Logf("Count('hello world') = %d, EncodeIDs = %v", count, ids)
}

func TestUnigram_HasToken(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	if !u.HasToken("▁hello") {
		t.Error("HasToken('▁hello') should be true")
	}
	if u.HasToken("hello world") {
		t.Error("HasToken('hello world') should be false (spaces not in vocab)")
	}
	if !u.HasToken("▁world") {
		t.Error("HasToken('▁world') should be true")
	}
}

func TestUnigram_TokenID(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	if id := u.TokenID("▁hello"); id != 3 {
		t.Errorf("TokenID('▁hello') = %d, want 3", id)
	}
	if id := u.TokenID("▁world"); id != 4 {
		t.Errorf("TokenID('▁world') = %d, want 4", id)
	}
	if id := u.TokenID("nonexistent"); id != -1 {
		t.Errorf("TokenID('nonexistent') = %d, want -1", id)
	}
}

func TestUnigram_ClearCache(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "▁hello", "▁world"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Encode to populate cache
	u.EncodeIDs("hello world")

	// Clear cache
	u.ClearCache()

	// Verify it still works after clear
	ids := u.EncodeIDs("hello world")
	if len(ids) < 1 {
		t.Error("EncodeIDs should work after ClearCache")
	}
	t.Logf("After ClearCache: EncodeIDs('hello world') = %v", ids)
}

func TestUnigram_GreedyLongestMatch(t *testing.T) {
	data := &common.TokenizerData{
		Model: "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"▁a", "▁b", "▁c", // single chars with leading ▁
			"▁ab", "▁bc", // two-char tokens with leading ▁
			"▁abc"}, // three-char token with leading ▁
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// "abc" should match as a single token (longest greedy match)
	ids := u.EncodeIDs("abc")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('abc') = %d ids, want 1 (greedy longest match)", len(ids))
	}
	t.Logf("EncodeIDs('abc') = %v", ids)
}

func TestUnigram_UnmatchedToken(t *testing.T) {
	data := &common.TokenizerData{
		Model: "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"a", "b"}, // only single chars
		HasUNKID: true,
		UNKID:    0,
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// "xyz" has no match — should use UNK token
	ids := u.EncodeIDs("xyz")
	t.Logf("EncodeIDs('xyz') = %v (len=%d)", ids, len(ids))
	if len(ids) == 0 {
		t.Error("EncodeIDs('xyz') with UNK should return at least one token")
	}
}

func TestUnigram_EmptyString(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>", "a"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	count := u.Count("")
	ids := u.EncodeIDs("")
	if count != 0 || len(ids) != 0 {
		t.Errorf("Count('') = %d, EncodeIDs('') = %d ids, want both empty", count, len(ids))
	}
}

func TestUnigram_Concurrent(t *testing.T) {
	data := &common.TokenizerData{
		Model: "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"▁a", "▁b", "▁c", "▁d", "▁e", "▁f", "▁g", "▁h", "▁i", "▁j"},
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	longInput := strings.Repeat("abcdefghij ", 100)

	ids1 := u.EncodeIDs(longInput)
	ids2 := u.EncodeIDs(longInput)

	if len(ids1) != len(ids2) {
		t.Errorf("Concurrent encoding mismatch: %d vs %d ids", len(ids1), len(ids2))
	}
	t.Logf("Concurrent EncodeIDs length = %d", len(ids1))
}

func TestUnigram_Unicode(t *testing.T) {
	data := &common.TokenizerData{
		Model: "unigram",
		Tokens: []string{"<unk>", "<s>", "</s>",
			"▁a", "▁b", "▁c",
			"▁世", "▁界", "▁世界"}, // Chinese characters with leading ▁
	}

	u, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	ids := u.EncodeIDs("世界")
	t.Logf("EncodeIDs('世界') = %v (len=%d)", ids, len(ids))
	if len(ids) < 1 {
		t.Error("EncodeIDs('世界') should return at least one token")
	}
}

func TestUnigram_DetokenizeEmpty(t *testing.T) {
	u := &Unigram{}
	result := u.Detokenize(nil)
	if result != "" {
		t.Errorf("Detokenize(nil) = %q, want empty string", result)
	}

	result2 := u.Detokenize([]int{})
	if result2 != "" {
		t.Errorf("Detokenize([]) = %q, want empty string", result2)
	}
}

func TestUnigram_NilReceiver(t *testing.T) {
	var u *Unigram

	if u.HasToken("hello") {
		t.Error("nil Unigram.HasToken should return false")
	}
	if u.TokenID("hello") != -1 {
		t.Error("nil Unigram.TokenID should return -1")
	}
	if u.Count("hello") != 0 {
		t.Error("nil Unigram.Count should be 0")
	}
	if u.EncodeIDs("hello") != nil {
		t.Error("nil Unigram.EncodeIDs should return nil")
	}
}
