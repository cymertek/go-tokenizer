package spm_test

import (
	"fmt"

	"github.com/cymertek/go-tokenizer/models/common"
	"github.com/cymertek/go-tokenizer/models/spm"
)

// ExampleNew demonstrates creating an SPM tokenizer from programmatic data.
func ExampleNew() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, err := spm.New(data)
	if err != nil {
		panic(err)
	}

	ids := s.EncodeIDs("hello world")
	fmt.Printf("Tokens: %v\n", ids)
	// Output: Tokens: [0 1]
}

// ExampleSPM_EncodeIDs demonstrates encoding text with SentencePiece tokenization.
func ExampleSPM_EncodeIDs() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	ids := s.EncodeIDs("hello world")
	fmt.Printf("Encoded %q → %v\n", "hello world", ids)
	// Output: Encoded "hello world" → [0 1]
}

// ExampleSPM_Detokenize demonstrates converting token IDs back to text.
func ExampleSPM_Detokenize() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	ids := []int{0, 1}
	text := s.Detokenize(ids)
	fmt.Printf("Decoded %v → %q\n", ids, text)
	// Output: Decoded [0 1] → "hello world"
}

// ExampleSPM_Count demonstrates counting tokens in text.
func ExampleSPM_Count() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	count := s.Count("hello world")
	fmt.Printf("Token count for %q: %d\n", "hello world", count)
	// Output: Token count for "hello world": 2
}

// ExampleSPM_HasToken demonstrates checking if a token exists.
func ExampleSPM_HasToken() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	fmt.Printf("Has '▁hello': %v\n", s.HasToken("▁hello"))
	fmt.Printf("Has 'xyz': %v\n", s.HasToken("xyz"))
	// Output: Has '▁hello': true
	// Has 'xyz': false
}

// ExampleSPM_TokenID demonstrates looking up a token's ID.
func ExampleSPM_TokenID() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	id := s.TokenID("world")
	fmt.Printf("'world' ID: %d\n", id)
	// Output: 'world' ID: 1
}

// ExampleSPM_Type demonstrates getting the tokenizer type.
func ExampleSPM_Type() {
	tokens := []string{"▁hello"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	fmt.Printf("Type: %s\n", s.Type())
	// Output: Type: spm
}

// ExampleSPM_IDToToken demonstrates converting an ID back to a token string.
func ExampleSPM_IDToToken() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	tok := s.IDToToken(0)
	fmt.Printf("ID 0 → %q\n", tok)
	// Output: ID 0 → "▁hello"
}

// ExampleSPM_ClearCache demonstrates clearing the encoding cache.
func ExampleSPM_ClearCache() {
	tokens := []string{"▁hello", "world"}
	data := &common.TokenizerData{Model: "spm", Tokens: tokens}

	s, _ := spm.New(data)
	// Encode some text to populate cache
	s.EncodeIDs("hello world")
	s.ClearCache()
	fmt.Println("Cache cleared")
	// Output: Cache cleared
}
