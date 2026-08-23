package bpe

import (
	"fmt"
	"strings"
	"testing"

	"github.com/cymertek/go-tokenizer/models/common"
)

// getTestData returns a BPE tokenizer with trie built.
func newTestBPE() *BPE {
	tokens := []string{
		"<unk>", "<s>", "</s>",
		"hello", "world", "hello world",
		"hel", "lo", "wor", "ld",
		"playing", "play", "ing", "played", "playe",
		"walking", "walk", "ing",
		"running", "run", "ning",
		"test", "testing", "tests",
	}
	b, _ := New(&common.TokenizerData{
		Tokens:   tokens,
		Merges:   nil,
		BOSID:    1,
		EOSID:    2,
		HasBOSID: true,
		HasEOSID: true,
		HasUNKID: true,
		UNKID:    0,
	})
	return b
}

func generateInputs(n int) []string {
	inputs := make([]string, n)
	words := []string{"hello", "world", "playing", "walk", "run", "test", "ing", "ed", "s", "ing"}
	for i := range inputs {
		nw := (i % 8) + 1
		for j := 0; j < nw; j++ {
			w := words[(i+j)%len(words)]
			if j > 0 {
				inputs[i] += " "
			}
			inputs[i] += w
		}
	}
	return inputs
}

func BenchmarkEncodeIDs_BPE(b *testing.B) {
	bpe := newTestBPE()
	inputs := generateInputs(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bpe.EncodeIDs(inputs[i%len(inputs)])
	}
}

func BenchmarkEncodeCount(b *testing.B) {
	bpe := newTestBPE()
	inputs := generateInputs(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bpe.Count(inputs[i%len(inputs)])
	}
}

func BenchmarkEncodeIDsVsCount(b *testing.B) {
	bpe := newTestBPE()
	inputs := generateInputs(100)
	b.Run("EncodeIDs", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = bpe.EncodeIDs(inputs[i%len(inputs)])
		}
	})
	b.Run("Count", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = bpe.Count(inputs[i%len(inputs)])
		}
	})
}

func BenchmarkEncodeIDs_BPE_Cache(b *testing.B) {
	bpe := newTestBPE()
	bpe.SetCache(1000)
	inputs := generateInputs(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bpe.EncodeIDs(inputs[i%len(inputs)])
	}
}

func BenchmarkEncodeIDs_VaryingVocabSize(b *testing.B) {
	for _, vocabSize := range []int{100, 500, 1000, 5000, 10000} {
		b.Run(fmt.Sprintf("vocab%d", vocabSize), func(b *testing.B) {
			tokens := make([]string, vocabSize)
			for i := range tokens {
				tokens[i] = fmt.Sprintf("tok%d", i)
			}
			bpe, _ := New(&common.TokenizerData{
				Tokens:   tokens,
				HasUNKID: true,
				UNKID:    0,
			})
			input := strings.Repeat("tok999 ", 100)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = bpe.EncodeIDs(input)
			}
		})
	}
}

func BenchmarkEncodeCount_Empty(b *testing.B) {
	bpe := newTestBPE()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bpe.Count("")
	}
}

func BenchmarkCompare_BPE(b *testing.B) {
	bpe := newTestBPE()
	input := "hello world playing walking running test"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bpe.EncodeIDs(input)
	}
}

func BenchmarkCompare_CountVsIDs(b *testing.B) {
	bpe := newTestBPE()
	input := "hello world playing walking running test"
	b.Run("Count", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bpe.Count(input)
		}
	})
	b.Run("IDs", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = bpe.EncodeIDs(input)
		}
	})
}

// BenchmarkEncodeTrie_Direct measures trie-only path (no map fallback).
func BenchmarkEncodeTrie_Direct(b *testing.B) {
	tokens := []string{
		"<unk>", "<s>", "</s>",
		"hello", "world", "hello world",
		"hel", "lo", "wor", "ld",
		"playing", "play", "ing", "played", "playe",
		"walking", "walk", "ing",
		"running", "run", "ning",
		"test", "testing", "tests",
	}
	bpe, _ := New(&common.TokenizerData{
		Tokens:   tokens,
		HasUNKID: true,
		UNKID:    -1,
	})
	input := "hello world playing walking running test"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bpe.EncodeIDs(input)
	}
}
