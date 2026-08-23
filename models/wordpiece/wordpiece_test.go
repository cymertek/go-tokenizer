package wordpiece

import (
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestWordPiece_NewProgrammatic(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}
	if w == nil {
		t.Fatal("New returned nil")
	}
	if w.Type() != "wordpiece" {
		t.Errorf("Type() = %q, want %q", w.Type(), "wordpiece")
	}
}

func TestWordPiece_NilData(t *testing.T) {
	w := &WordPiece{}
	if w == nil {
		t.Fatal("nil WordPiece should not be nil")
	}
	if w.Count("") != 0 {
		t.Error("Count('') on nil WordPiece should be 0")
	}
	if len(w.EncodeIDs("")) != 0 {
		t.Error("EncodeIDs('') on nil WordPiece should return empty slice")
	}
}

func TestWordPiece_EncodeIDs(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Tokenize "playing" — should find "play" then "##ing".
	ids := w.EncodeIDs("playing")
	t.Logf("EncodeIDs('playing') = %v (len=%d)", ids, len(ids))
	if len(ids) < 1 {
		t.Errorf("EncodeIDs('playing') should return at least 1 token, got %d", len(ids))
	}

	// Tokenize "hello world" — two separate words.
	ids2 := w.EncodeIDs("hello world")
	t.Logf("EncodeIDs('hello world') = %v (len=%d)", ids2, len(ids2))
	if len(ids2) < 1 {
		t.Errorf("EncodeIDs('hello world') should return at least 1 token, got %d", len(ids2))
	}

	// Tokenize "play" — direct vocab match.
	ids3 := w.EncodeIDs("play")
	t.Logf("EncodeIDs('play') = %v (len=%d)", ids3, len(ids3))
	if len(ids3) < 1 {
		t.Errorf("EncodeIDs('play') should return at least 1 token, got %d", len(ids3))
	}
}

func TestWordPiece_Detokenize(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// Encode then decode — should recover original text.
	ids := w.EncodeIDs("playing")
	decoded := w.Detokenize(ids)
	t.Logf("Detokenize(%v) = %q", ids, decoded)

	if !strings.Contains(decoded, "play") || !strings.Contains(decoded, "ing") {
		t.Errorf("Detokenize should contain 'play' and 'ing', got %q", decoded)
	}

	// Test multi-word.
	ids2 := w.EncodeIDs("hello world")
	decoded2 := w.Detokenize(ids2)
	t.Logf("Detokenize(%v) = %q", ids2, decoded2)

	if !strings.Contains(decoded2, "hello") || !strings.Contains(decoded2, "world") {
		t.Errorf("Detokenize should contain 'hello' and 'world', got %q", decoded2)
	}
}

func TestWordPiece_Count(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	count := w.Count("playing")
	ids := w.EncodeIDs("playing")
	if count != len(ids) {
		t.Errorf("Count('playing') = %d, len(EncodeIDs) = %d", count, len(ids))
	}
	t.Logf("Count('playing') = %d, EncodeIDs = %v", count, ids)
}

func TestWordPiece_HasToken(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	if !w.HasToken("play") {
		t.Error("HasToken('play') should be true")
	}
	if !w.HasToken("##ing") {
		t.Error("HasToken('##ing') should be true")
	}
	if w.HasToken("playing") {
		t.Error("HasToken('playing') should be false (not in vocab)")
	}
}

func TestWordPiece_TokenID(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	if id := w.TokenID("play"); id != 3 {
		t.Errorf("TokenID('play') = %d, want 3", id)
	}
	if id := w.TokenID("##ing"); id != 4 {
		t.Errorf("TokenID('##ing') = %d, want 4", id)
	}
	if id := w.TokenID("nonexistent"); id != -1 {
		t.Errorf("TokenID('nonexistent') = %d, want -1", id)
	}
}

func TestWordPiece_ClearCache(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "play", "##ing"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	w.EncodeIDs("playing")
	w.ClearCache()

	ids := w.EncodeIDs("playing")
	if len(ids) < 1 {
		t.Error("EncodeIDs should work after ClearCache")
	}
	t.Logf("After ClearCache: EncodeIDs('playing') = %v", ids)
}

func TestWordPiece_UnmatchedToken(t *testing.T) {
	data := &common.TokenizerData{
		Model: "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]",
			"pl", "##ay"}, // only subwords, no full word
		HasUNKID: true,
		UNKID:    0,
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	// "xyz" has no match — should use UNK token.
	ids := w.EncodeIDs("xyz")
	t.Logf("EncodeIDs('xyz') = %v (len=%d)", ids, len(ids))
	if len(ids) == 0 {
		t.Error("EncodeIDs('xyz') with UNK should return at least one token")
	}
}

func TestWordPiece_EmptyString(t *testing.T) {
	data := &common.TokenizerData{
		Model:  "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]", "a"},
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	count := w.Count("")
	ids := w.EncodeIDs("")
	if count != 0 || len(ids) != 0 {
		t.Errorf("Count('') = %d, EncodeIDs('') = %d ids, want both empty", count, len(ids))
	}
}

func TestWordPiece_Unicode(t *testing.T) {
	data := &common.TokenizerData{
		Model: "wordpiece",
		Tokens: []string{"[UNK]", "[CLS]", "[SEP]",
			"a", "b", "c",
			"世", "界", "##世", "##界"}, // Chinese characters with continuation
	}

	w, err := New(data)
	if err != nil {
		t.Fatalf("New error: %v", err)
	}

	ids := w.EncodeIDs("世界")
	t.Logf("EncodeIDs('世界') = %v (len=%d)", ids, len(ids))
	if len(ids) < 1 {
		t.Error("EncodeIDs('世界') should return at least one token")
	}
}

func TestWordPiece_DetokenizeEmpty(t *testing.T) {
	w := &WordPiece{}
	result := w.Detokenize(nil)
	if result != "" {
		t.Errorf("Detokenize(nil) = %q, want empty string", result)
	}

	result2 := w.Detokenize([]int{})
	if result2 != "" {
		t.Errorf("Detokenize([]) = %q, want empty string", result2)
	}
}

func TestWordPiece_NilReceiver(t *testing.T) {
	var w *WordPiece

	if w.HasToken("hello") {
		t.Error("nil WordPiece.HasToken should return false")
	}
	if w.TokenID("hello") != -1 {
		t.Error("nil WordPiece.TokenID should return -1")
	}
	if w.Count("hello") != 0 {
		t.Error("nil WordPiece.Count should be 0")
	}
	if w.EncodeIDs("hello") != nil {
		t.Error("nil WordPiece.EncodeIDs should return nil")
	}
}
