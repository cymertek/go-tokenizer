package bpe

import (
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

func TestPreTokenize_Qwen2(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"Hello world", []string{"Hello", "world"}},
		{"Hello", []string{"Hello"}},
		{"it's a test", []string{"it", "'s", "a", "test"}},
		{"  spaces  ", []string{"spaces"}},
	}
	for _, tc := range tests {
		got := preTokenize(tc.input, PreQwen2, 0)
		var texts []string
		for _, s := range got {
			texts = append(texts, s.Text)
		}
		if len(texts) != len(tc.want) {
			t.Errorf("preTokenize(%q) = %d fragments, want %d: %v", tc.input, len(texts), len(tc.want), texts)
			continue
		}
		for i := range texts {
			if texts[i] != tc.want[i] {
				t.Errorf("preTokenize(%q)[%d] = %q, want %q", tc.input, i, texts[i], tc.want[i])
			}
		}
	}
}

func TestPreTokenize_GPT2(t *testing.T) {
	got := preTokenize("Hello world", PreGPT2, 0)
	var texts []string
	for _, s := range got {
		texts = append(texts, s.Text)
	}
	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != "world" {
		t.Errorf("preTokenize('Hello world', PreGPT2) = %v, want [Hello world]", texts)
	}
}

func TestPreTokenize_Llama3(t *testing.T) {
	got := preTokenize("Hello world", PreLlama3, 0)
	var texts []string
	for _, s := range got {
		texts = append(texts, s.Text)
	}
	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != "world" {
		t.Errorf("preTokenize('Hello world', PreLlama3) = %v, want [Hello world]", texts)
	}
}

func TestPreTokenize_Gemma4_NoSplit(t *testing.T) {
	got := preTokenize("hello world", PreGemma4, 0)
	if len(got) != 1 || got[0].Text != "hello world" {
		t.Errorf("preTokenize('hello world', PreGemma4) = %v, want [hello world]", got)
	}
}

func TestPreTokenize_Contractions(t *testing.T) {
	tests := []struct {
		input string
		typ   PreType
	}{
		{"don't you know?", PreQwen2},
		{"I'm glad it's working", PreQwen2},
		{"haven't we?", PreGPT2},
	}
	for _, tc := range tests {
		got := preTokenize(tc.input, tc.typ, 0)
		if len(got) == 0 {
			t.Errorf("preTokenize(%q) = 0 fragments", tc.input)
		}
		for _, s := range got {
			if s.Text == "'s" || s.Text == "'t" || s.Text == "'re" || s.Text == "'ve" || s.Text == "'m" || s.Text == "'ll" || s.Text == "'d" {
				t.Logf("found contraction %q in %v", s.Text, got)
			}
		}
	}
}

func TestPreTokenize_Empty(t *testing.T) {
	if got := preTokenize("", PreQwen2, 0); len(got) != 0 {
		t.Errorf("preTokenize('') = %v, want empty", got)
	}
}

func TestPreTokenize_Number(t *testing.T) {
	got := preTokenize("I have 42 apples and 123 oranges", PreQwen2, 0)
	var texts []string
	for _, s := range got {
		texts = append(texts, s.Text)
	}
	found := false
	for _, t := range texts {
		if t == "42" || t == "123" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("preTokenize('I have 42 apples and 123 oranges') = %v, want to keep numbers intact", texts)
	}
}

func TestPreSplit_HasSpace(t *testing.T) {
	ps := PreSplit{Text: " world"}
	if !ps.HasSpace() {
		t.Error("HasSpace(' world') should be true")
	}
	ps2 := PreSplit{Text: "world"}
	if ps2.HasSpace() {
		t.Error("HasSpace('world') should be false")
	}
	ps3 := PreSplit{Text: "\tindented"}
	if !ps3.HasSpace() {
		t.Error("HasSpace(tab-indented) should be true")
	}
}

func TestPreSplit_Trimspace(t *testing.T) {
	original := PreSplit{Text: " world "}
	trimmed := original.TrimSpace()
	if trimmed.Text != "world" {
		t.Errorf("TrimSpace(' world ') = %q, want %q", trimmed.Text, "world")
	}
}

func TestBPE_PreTypeQwen2(t *testing.T) {
	data := &common.TokenizerData{
		Tokens:  []string{"hello", "Ġworld", "hello world"},
		Merges:  []common.Merge{},
		PreType: PreQwen2,
	}
	bpe, _ := New(data)

	ids := bpe.EncodeIDs("hello world")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('hello world') = %d ids, want 1: %v", len(ids), ids)
	}
}

func TestBPE_PreTypeGemma4(t *testing.T) {
	data := &common.TokenizerData{
		Tokens:  []string{"hello", " world", "hello world"},
		Merges:  []common.Merge{},
		PreType: PreGemma4,
	}
	bpe, _ := New(data)

	ids := bpe.EncodeIDs("hello world")
	if len(ids) != 1 {
		t.Errorf("EncodeIDs('hello world', PreGemma4) = %d ids, want 1", len(ids))
	}
}

func TestEncodeCount_PreTypeConsistency(t *testing.T) {
	pres := []PreType{PreQwen2, PreGPT2, PreLlama3, PreGemma4, PreDefault}
	inputs := []string{"hello world", "test", "  spaced  ", "a-b"}
	for _, pt := range pres {
		for _, input := range inputs {
			bpe, _ := New(&common.TokenizerData{
				Tokens:  []string{"hello", " world", "hel", "lo", "wor", "ld", "a", "-b"},
				Merges:  []common.Merge{},
				PreType: pt,
			})
			ids := bpe.EncodeIDs(input)
			count := bpe.Count(input)
			if input != "" && count != len(ids) {
				t.Errorf("[%d] Count(%q) = %d, len(EncodeIDs) = %d",
					pt, input, count, len(ids))
			}
		}
	}
}
