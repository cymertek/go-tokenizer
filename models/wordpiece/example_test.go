package wordpiece_test

import (
	"fmt"

	"github.com/cymertek/go-tokenizer/models/common"
	"github.com/cymertek/go-tokenizer/models/wordpiece"
)

// ExampleNew demonstrates creating a WordPiece tokenizer from programmatic data.
func ExampleNew() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, err := wordpiece.New(data)
	if err != nil {
		panic(err)
	}

	ids := w.EncodeIDs("hello world")
	fmt.Printf("Tokens: %v\n", ids)
	// Output: Tokens: [0 1]
}

// ExampleWordPiece_EncodeIDs demonstrates encoding text with WordPiece tokenization.
func ExampleWordPiece_EncodeIDs() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	ids := w.EncodeIDs("hello world")
	fmt.Printf("Encoded %q → %v\n", "hello world", ids)
	// Output: Encoded "hello world" → [0 1]
}

// ExampleWordPiece_Detokenize demonstrates converting token IDs back to text.
func ExampleWordPiece_Detokenize() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	ids := []int{0, 1}
	text := w.Detokenize(ids)
	fmt.Printf("Decoded %v → %q\n", ids, text)
	// Output: Decoded [0 1] → "hello world"
}

// Example demonstrates WordPiece subword tokenization with ## continuation.
func Example() {
	tokens := []string{"play", "##ing", "hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	ids := w.EncodeIDs("playing")
	fmt.Printf("Encoded 'playing' → %v\n", ids)
	text := w.Detokenize(ids)
	fmt.Printf("Decoded back to: %q\n", text)
	// Output: Encoded 'playing' → [0 1]
	// Decoded back to: "playing"
}

// ExampleWordPiece_Count demonstrates counting tokens in text.
func ExampleWordPiece_Count() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	count := w.Count("hello world")
	fmt.Printf("Token count for %q: %d\n", "hello world", count)
	// Output: Token count for "hello world": 2
}

// ExampleWordPiece_HasToken demonstrates checking if a token exists.
func ExampleWordPiece_HasToken() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	fmt.Printf("Has 'hello': %v\n", w.HasToken("hello"))
	fmt.Printf("Has 'xyz': %v\n", w.HasToken("xyz"))
	// Output: Has 'hello': true
	// Has 'xyz': false
}

// ExampleWordPiece_TokenID demonstrates looking up a token's ID.
func ExampleWordPiece_TokenID() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	id := w.TokenID("world")
	fmt.Printf("'world' ID: %d\n", id)
	// Output: 'world' ID: 1
}

// ExampleWordPiece_Type demonstrates getting the tokenizer type.
func ExampleWordPiece_Type() {
	tokens := []string{"hello"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	fmt.Printf("Type: %s\n", w.Type())
	// Output: Type: wordpiece
}

// ExampleWordPiece_IDToToken demonstrates converting an ID back to a token string.
func ExampleWordPiece_IDToToken() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	tok := w.IDToToken(0)
	fmt.Printf("ID 0 → %q\n", tok)
	// Output: ID 0 → "hello"
}

// ExampleWordPiece_ClearCache demonstrates clearing the encoding cache.
func ExampleWordPiece_ClearCache() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "wordpiece", Tokens: tokens}

	w, _ := wordpiece.New(data)
	// Encode some text to populate cache
	w.EncodeIDs("hello world")
	w.ClearCache()
	fmt.Println("Cache cleared")
	// Output: Cache cleared
}
