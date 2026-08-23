package models

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestNewBPE(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"<unk>", "<s>", "</s>", "hello", "world", "hello world"},
		Merges: []Merge{
			{A: "hel", B: "lo"},
			{A: "wor", B: "ld"},
		},
		BOSID: 1,
		EOSID: 2,
		UNKID: 0,
	}
	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}
	if bpe.Type() != "bpe" {
		t.Errorf("Type() = %q, want %q", bpe.Type(), "bpe")
	}
}

func TestNewBPENilData(t *testing.T) {
	bpe, _ := NewBPE(nil)
	if bpe == nil {
		t.Fatal("NewBPE(nil) should not panic")
	}
}

func TestNewBPENilTokens(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string(nil),
		Merges: []Merge{},
		UNKID:  0,
	}
	bpe, _ := NewBPE(data)
	if bpe == nil {
		t.Fatal("NewBPE with nil tokens should not panic")
	}
	count := bpe.Count("")
	if count != 0 {
		t.Errorf("Count('') = %d, want 0", count)
	}
}

func TestEncodeIDs(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"<unk>", "hello", "world", "hello world", "h", "he", "hel"},
		Merges:   []Merge{},
		BOSID:    -1,
		EOSID:    -1,
		HasBOSID: true,
		HasEOSID: true,
		HasUNKID: true,
		UNKID:    0,
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("hello world")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('hello world') = %d ids, want 1 (direct vocab match)", len(ids))
	}
}

func TestEncodeIDsWithBOS_EOS(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"<unk>", "hello", "world", "hello world"},
		Merges:   []Merge{},
		BOSID:    100,
		EOSID:    101,
		HasBOSID: true,
		HasEOSID: true,
		HasUNKID: true,
		UNKID:    0,
		AddBOS:   true,
		AddEOS:   true,
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("hello world")
	if len(ids) != 3 {
		t.Errorf("EncodeIDs('hello world') with BOS/EOS = %d ids, want 3", len(ids))
	}
	if ids[0] != 100 {
		t.Errorf("first id = %d, want %d (BOS)", ids[0], 100)
	}
	if ids[len(ids)-1] != 101 {
		t.Errorf("last id = %d, want %d (EOS)", ids[len(ids)-1], 101)
	}
}

func TestEncodeIDSEmpty(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a", "b"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)
	ids := bpe.EncodeIDs("")
	if len(ids) != 0 {
		t.Errorf("EncodeIDs('') = %d ids, want 0", len(ids))
	}
}

func TestEncodeCount(t *testing.T) {
	allTokens := []string{"<unk>", "<s>", "</s>"}
	words := []string{"hello", "world", "hel", "lo", "wor", "ld", "hello world"}
	allTokens = append(allTokens, words...)

	bpe, _ := NewBPE(&TokenizerData{
		Tokens:   allTokens,
		Merges:   []Merge{{A: "hel", B: "lo"}, {A: "wor", B: "ld"}},
		BOSID:    1,
		EOSID:    2,
		HasBOSID: true,
		HasEOSID: true,
		HasUNKID: true,
		UNKID:    0,
	})

	inputs := []string{
		"hello world",
		"  multiple   spaces  ",
		"a",
		" a b c ",
		"abc",
		"",
	}
	for _, input := range inputs {
		ids := bpe.EncodeIDs(input)
		count := bpe.Count(input)
		if input != "" && count != len(ids) {
			t.Errorf("Inconsistency for %q: EncodeIDs=%d ids, Count=%d",
				input, len(ids), count)
		}
	}
}

func TestEncodeCount_NoBOS_EOS(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "world", "hello world"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)

	count := bpe.Count("hello world")
	ids := bpe.EncodeIDs("hello world")
	if count != len(ids) {
		t.Errorf("Count('hello world') = %d, EncodeIDs len = %d", count, len(ids))
	}
}

func TestEncodeCount_EmptyString(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)
	if bpe.Count("") != 0 {
		t.Error("Count('') should be 0")
	}
}

func TestEncodeIDSGreedyLongestMatch(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"a", "ab", "abc", "b", "bc", "c"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("abc")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('abc') = %d ids, want 1 (longest match)", len(ids))
	}
	_ = ids
}

func TestEncodeIDSCache(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a", "b", "ab"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)
	bpe.SetCache(10)

	ids1 := bpe.EncodeIDs("ab")
	ids2 := bpe.EncodeIDs("ab")
	if len(ids1) != len(ids2) {
		t.Errorf("Cached vs uncached: %d vs %d", len(ids1), len(ids2))
	}
}

func TestClearCache(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"a", "b"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)
	bpe.SetCache(10)
	bpe.EncodeIDs("a")
	bpe.ClearCache()

	bpe.SetCache(10)
	ids := bpe.EncodeIDs("a")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('a') after clear = %d ids, want 1", len(ids))
	}
}

func TestModelType(t *testing.T) {
	bpe, _ := NewBPE(&TokenizerData{Tokens: []string{"a"}})
	if bpe.Type() != "bpe" {
		t.Errorf("Type() = %q, want %q", bpe.Type(), "bpe")
	}
}

func TestBPE_NilVocab(t *testing.T) {
	bpe := &BPE{}
	ids := bpe.EncodeIDs("hello")
	t.Logf("EncodeIDs on nil vocab = %v", ids)
}

func TestBPE_NoUNKAvaliable(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hel", "lo"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)
	ids := bpe.EncodeIDs("xyz")
	if len(ids) != 0 {
		t.Errorf("EncodeIDs('xyz') with no UNK = %v, want nil or empty", ids)
	}
}

func TestBPE_Tokens(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"<unk>", "hello", "world", "hello world"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	tokens := bpe.Tokens("hello world")
	if len(tokens) != 1 || tokens[0] != "hello world" {
		t.Errorf("Tokens('hello world') = %v, want [hello world]", tokens)
	}
}

func TestEncodeCount_WhitespaceOnly(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a", "b"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)
	if bpe.Count("   ") != 0 {
		t.Error("Count('   ') should be 0")
	}
}

func TestEncodeCount_MultiWord(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"a", "b", "c", "ab", "bc", "abc"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	count := bpe.Count("ab bc")
	if count != 2 {
		t.Errorf("Count('ab bc') = %d, want 2", count)
	}
}

func TestBPE_ConsecutiveWhitespace(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"hello", "world", "  ", "hello  world"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("hello  world")
	if len(ids) < 1 {
		t.Error("EncodeIDs should return at least one token")
	}
	t.Logf("EncodeIDs('hello  world') = %v (len=%d)", ids, len(ids))
}

func TestBPE_Deterministic(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a", "b", "ab"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)

	ids1 := bpe.EncodeIDs("ab")
	ids2 := bpe.EncodeIDs("ab")
	count3 := bpe.Count("ab")

	if len(ids1) != len(ids2) {
		t.Error("EncodeIDs should be deterministic")
	}
	if count3 != len(ids1) {
		t.Errorf("Count vs EncodeIDs inconsistent: %d vs %d", count3, len(ids1))
	}
}

func TestBPE_TokenizesLongWord(t *testing.T) {
	var tokens []string
	for i := 0; i < 256; i++ {
		tokens = append(tokens, string(rune(i)))
	}
	tokens = append(tokens, "th", "he", "in", "er", "an", "on", "en", "at", "to", "ed")
	tokens = append(tokens, "the", "ing", "ion", "tion")

	data := &TokenizerData{
		Tokens: tokens,
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)

	inputs := []string{"the", "ing", "theing", "testing"}
	for _, input := range inputs {
		ids := bpe.EncodeIDs(input)
		count := bpe.Count(input)
		if count != len(ids) {
			t.Errorf("Inconsistency for %q: IDs=%d, Count=%d", input, len(ids), count)
		}
		t.Logf("EncodeIDs(%q) = %v (count=%d)", input, ids, count)
	}
}

func TestBPE_EmptyWord(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"a", "b"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)
	ids := bpe.EncodeIDs("a b")
	if len(ids) != 2 {
		t.Errorf("EncodeIDs('a b') = %d ids, want 2", len(ids))
	}
}

func TestBPE_GoldenPath(t *testing.T) {
	tokens := []string{
		"<unk>", "<s>", "</s>",
		"hello", "world", "hello world",
		"hel", "lo", "wor", "ld", "h", "he", "hell",
		"worl", "world",
		"pl", "la", "pla", "play", "ing", "playing",
	}
	bpe, _ := NewBPE(&TokenizerData{
		Tokens:   tokens,
		Merges:   []Merge{{A: "pla", B: "y"}, {A: "play", B: "ing"}},
		HasUNKID: true,
		UNKID:    0,
	})

	ids := bpe.EncodeIDs("hello world")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('hello world') = %d ids, want 1", len(ids))
	}

	ids2 := bpe.EncodeIDs("playing")
	if len(ids2) != 1 {
		t.Errorf("EncodeIDs('playing') = %d ids, want 1", len(ids2))
	}

	t.Logf("hello world tokens = %v, playing tokens = %v", ids, ids2)
}

func TestBPE_UnicodeWords(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hello", "世界", "hello世界", "世", "界"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("世界")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('世界') = %d ids, want 1", len(ids))
	}
	_ = ids
}

func TestBPE_SpaceInMiddle(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hello ", " world", "hello world", "hello"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("hello world")
	t.Logf("EncodeIDs('hello world') = %v (len=%d)", ids, len(ids))
	_ = ids
}

func TestBPE_SameTokenID(t *testing.T) {
	data := &TokenizerData{
		Tokens: []string{"a", "b", "c"},
		Merges: []Merge{},
	}
	bpe, _ := NewBPE(data)
	count := bpe.Count("a")
	if count != 1 {
		t.Errorf("Count('a') = %d, want 1", count)
	}
}

func TestBPE_VeryLongWord(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"a", "b"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	longWord := strings.Repeat("ab", 1000)
	ids := bpe.EncodeIDs(longWord)
	count := bpe.Count(longWord)
	if count != len(ids) {
		t.Errorf("Inconsistency for long word: IDs=%d, Count=%d", len(ids), count)
	}
}

func TestBPE_NewlineAndTab(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"hello", " world", "hello world"},
		Merges:   []Merge{},
		PreType:  PreQwen2,
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	ids := bpe.EncodeIDs("hello world")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('hello world') = %d ids, want 1: %v", len(ids), ids)
	}
}

func TestHasToken(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"<unk>", "hello", "world", "hello world"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	if !bpe.HasToken("hello") {
		t.Error("HasToken('hello') should be true")
	}
	if bpe.HasToken("xyz") {
		t.Error("HasToken('xyz') should be false")
	}
	if !bpe.HasToken("hello world") {
		t.Error("HasToken('hello world') should be true")
	}
}

func TestTokenID(t *testing.T) {
	data := &TokenizerData{
		Tokens:   []string{"<unk>", "hello", "world", "hello world"},
		Merges:   []Merge{},
		HasUNKID: true,
		UNKID:    -1,
	}
	bpe, _ := NewBPE(data)

	if id := bpe.TokenID("hello"); id != 1 {
		t.Errorf("TokenID('hello') = %d, want 1", id)
	}
	if id := bpe.TokenID("world"); id != 2 {
		t.Errorf("TokenID('world') = %d, want 2", id)
	}
	if id := bpe.TokenID("xyz"); id != -1 {
		t.Errorf("TokenID('xyz') = %d, want -1", id)
	}
}

func TestBPE_NilEncode(t *testing.T) {
	var bpe *BPE
	ids := bpe.EncodeIDs("hello")
	if ids != nil {
		t.Errorf("nil BPE.EncodeIDs = %v, want nil", ids)
	}
	if bpe.Count("hello") != 0 {
		t.Error("nil BPE.Count should be 0")
	}
}

// ---- Pre-tokenization tests (moved to models/bpe package for internal access) ----

func TestBPE_PreTypeFromString(t *testing.T) {
	tests := []struct {
		input  string
		expect PreType
	}{
		{"qwen2", PreQwen2},
		{"megrez", PreQwen2},
		{"gpt-2", PreGPT2},
		{"phi-2", PreGPT2},
		{"jina-es", PreGPT2},
		{"llama3", PreLlama3},
		{"llama-v3", PreLlama3},
		{"llama-bpe", PreLlama3},
		{"falcon3", PreLlama3},
		{"pixtral", PreLlama3},
		{"starcoder", PreStarcoder},
		{"refact", PreStarcoder},
		{"command-r", PreStarcoder},
		{"smollm", PreStarcoder},
		{"deepseek-llm", PreDeepSeekLLM},
		{"deepseek-coder", PreDeepSeekLLM},
		{"deepseek-v3", PreDeepSeekLLM},
		{"falcon", PreFalcon},
		{"qwen35", PreQwen35},
		{"stablelm2", PreStableLM2},
		{"hunyuan", PreStableLM2},
		{"gpt-4o", PreGPT4O},
		{"llama4", PreGPT4O},
		{"gemma4", PreGemma4},
		{"mpt", PreGPT2},
		{"olmo", PreGPT2},
		{"jais", PreGPT2},
		{"", PreDefault},
		{"unknown", PreDefault},
	}
	for _, tc := range tests {
		got := common.PreTypeFromString(tc.input)
		if got != tc.expect {
			t.Errorf("PreTypeFromString(%q) = %d, want %d", tc.input, got, tc.expect)
		}
	}
}

func TestBPE_ConcurrentThreshold(t *testing.T) {
	vocab := make([]string, 1000)
	for i := 0; i < 1000; i++ {
		vocab[i] = fmt.Sprintf("token%d", i)
	}
	for i := 0; i < 500; i++ {
		vocab = append(vocab, fmt.Sprintf(" Ġ%d", i))
	}

	bpe, _ := NewBPE(&TokenizerData{
		Tokens:  vocab,
		Merges:  []Merge{},
		PreType: PreQwen2,
	})

	longInput := strings.Repeat("token123 the quick brown fox jumps over the lazy dog ", 1200)

	ids1 := bpe.EncodeIDs(longInput)
	ids2 := bpe.EncodeIDs(longInput)

	if len(ids1) != len(ids2) {
		t.Errorf("Concurrent vs sequential: %d vs %d ids", len(ids1), len(ids2))
	}
}

func TestBPE_ConcurrentEncodingMatch(t *testing.T) {
	vocab := make([]string, 2000)
	vocab[0] = "eos"
	for i := 1; i <= 1999; i++ {
		vocab[i] = fmt.Sprintf("w%d", i)
	}
	for i := 1; i <= 500; i++ {
		vocab = append(vocab, fmt.Sprintf(" %s", fmt.Sprintf("w%d", i)))
	}

	bpe, _ := NewBPE(&TokenizerData{
		Tokens:  vocab,
		Merges:  []Merge{},
		PreType: PreQwen2,
	})

	longInput := strings.Repeat("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 ", 2000)

	bpe.ClearCache()
	ids1 := bpe.EncodeIDs(longInput)
	bpe.ClearCache()
	ids2 := bpe.EncodeIDs(longInput)

	if len(ids1) != len(ids2) {
		t.Errorf("Concurrent vs sequential length mismatch: %d vs %d", len(ids1), len(ids2))
		minLen := len(ids1)
		if len(ids2) < minLen {
			minLen = len(ids2)
		}
		for i := 0; i < minLen; i++ {
			if ids1[i] != ids2[i] {
				t.Errorf("First difference at token %d: %d vs %d", i, ids1[i], ids2[i])
				break
			}
		}
	}
}
