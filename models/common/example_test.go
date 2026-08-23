package common_test

import (
	"fmt"

	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// ExampleNewProgrammatic demonstrates creating a TokenizerData for programmatic use.
func ExampleNewProgrammatic() {
	tokens := []string{"hello", "world"}
	data := common.NewProgrammatic("spm", tokens)
	fmt.Printf("Model: %s, Tokens: %v\n", data.Model, data.Tokens)
	// Output: Model: spm, Tokens: [hello world]
}

// ExampleNew demonstrates creating a tokenizer from programmatic data.
func ExampleNew() {
	// Import bpe to trigger its init() registration
	_ = bpe.New

	data := common.NewProgrammatic("bpe", []string{"hello", "world"})
	tok, err := common.New(data)
	if err != nil {
		panic(err)
	}

	ids := tok.EncodeIDs("hello world")
	fmt.Printf("Encoded %d tokens: %v\n", len(ids), ids)
	// Output: Encoded 2 tokens: [0 1]
}

// ExampleRegisteredTypes demonstrates listing all registered tokenizer types.
func ExampleRegisteredTypes() {
	// Import bpe to trigger registration
	_ = bpe.New

	types := common.RegisteredTypes()
	fmt.Printf("Available types: %v\n", types)
	// Output: Available types: [bpe]
}
