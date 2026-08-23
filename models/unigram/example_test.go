package unigram_test

import (
	"fmt"

	"github.com/cymertek/go-tokenizer/models/common"
	"github.com/cymertek/go-tokenizer/models/unigram"
)

// ExampleNew demonstrates creating an Unigram tokenizer from programmatic data.
func ExampleNew() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, err := unigram.New(data)
	if err != nil {
		panic(err)
	}

	ids := u.EncodeIDs("hello world")
	fmt.Printf("Tokens: %v\n", ids)
	// Output: Tokens: [0 1]
}

// ExampleUnigram_EncodeIDs demonstrates encoding text with Unigram tokenization.
func ExampleUnigram_EncodeIDs() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	ids := u.EncodeIDs("hello world")
	fmt.Printf("Encoded %q → %v\n", "hello world", ids)
	// Output: Encoded "hello world" → [0 1]
}

// ExampleUnigram_Detokenize demonstrates converting token IDs back to text.
func ExampleUnigram_Detokenize() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	ids := []int{0, 1}
	text := u.Detokenize(ids)
	fmt.Printf("Decoded %v → %q\n", ids, text)
	// Output: Decoded [0 1] → "hello world"
}

// ExampleUnigram_Count demonstrates counting tokens in text.
func ExampleUnigram_Count() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	count := u.Count("hello world")
	fmt.Printf("Token count for %q: %d\n", "hello world", count)
	// Output: Token count for "hello world": 2
}

// ExampleUnigram_HasToken demonstrates checking if a token exists.
func ExampleUnigram_HasToken() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	fmt.Printf("Has '▁hello': %v\n", u.HasToken("▁hello"))
	fmt.Printf("Has 'xyz': %v\n", u.HasToken("xyz"))
	// Output: Has '▁hello': true
	// Has 'xyz': false
}

// ExampleUnigram_TokenID demonstrates looking up a token's ID.
func ExampleUnigram_TokenID() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	id := u.TokenID("▁world")
	fmt.Printf("'▁world' ID: %d\n", id)
	// Output: '▁world' ID: 1
}

// ExampleUnigram_Type demonstrates getting the tokenizer type.
func ExampleUnigram_Type() {
	tokens := []string{"▁hello"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	fmt.Printf("Type: %s\n", u.Type())
	// Output: Type: unigram
}

// ExampleUnigram_IDToToken demonstrates converting an ID back to a token string.
func ExampleUnigram_IDToToken() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	tok := u.IDToToken(0)
	fmt.Printf("ID 0 → %q\n", tok)
	// Output: ID 0 → "▁hello"
}

// ExampleUnigram_ClearCache demonstrates clearing the encoding cache.
func ExampleUnigram_ClearCache() {
	tokens := []string{"▁hello", "▁world"}
	data := &common.TokenizerData{Model: "unigram", Tokens: tokens}

	u, _ := unigram.New(data)
	// Encode some text to populate cache
	u.EncodeIDs("hello world")
	u.ClearCache()
	fmt.Println("Cache cleared")
	// Output: Cache cleared
}
