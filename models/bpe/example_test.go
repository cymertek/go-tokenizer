package bpe_test

import (
	"fmt"

	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// ExampleNew demonstrates creating a BPE tokenizer from programmatic data.
func ExampleNew() {
	tokens := []string{"hello", "world", "hello world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, err := bpe.New(data)
	if err != nil {
		panic(err)
	}

	ids := b.EncodeIDs("hello world")
	fmt.Printf("Tokens: %v\n", ids)
	// Output: Tokens: [2]
}

// ExampleBPE_EncodeIDs demonstrates encoding text into token IDs.
func ExampleBPE_EncodeIDs() {
	tokens := []string{"hel", "lo", "wor", "ld", "hello", "world"}
	merges := []common.Merge{{A: "hel", B: "lo"}, {A: "wor", B: "ld"}}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens, Merges: merges}

	b, _ := bpe.New(data)
	ids := b.EncodeIDs("hello world")
	fmt.Printf("Encoded %q → %v\n", "hello world", ids)
	// Output: Encoded "hello world" → [4 5]
}

// ExampleBPE_Detokenize demonstrates converting token IDs back to text.
func ExampleBPE_Detokenize() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	ids := []int{0, 1}
	text := b.Detokenize(ids)
	fmt.Printf("Decoded %v → %q\n", ids, text)
	// Output: Decoded [0 1] → "hello world"
}

// ExampleBPE_Count demonstrates counting tokens in text.
func ExampleBPE_Count() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	count := b.Count("hello world")
	fmt.Printf("Token count for %q: %d\n", "hello world", count)
	// Output: Token count for "hello world": 2
}

// ExampleBPE_Tokens demonstrates getting token strings.
func ExampleBPE_Tokens() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	tokenStrs := b.Tokens("hello world")
	fmt.Printf("Tokens: %v\n", tokenStrs)
	// Output: Tokens: [hello world]
}

// ExampleBPE_HasToken demonstrates checking if a token exists.
func ExampleBPE_HasToken() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	fmt.Printf("Has 'hello': %v\n", b.HasToken("hello"))
	fmt.Printf("Has 'xyz': %v\n", b.HasToken("xyz"))
	// Output: Has 'hello': true
	// Has 'xyz': false
}

// ExampleBPE_TokenID demonstrates looking up a token's ID.
func ExampleBPE_TokenID() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	id := b.TokenID("world")
	fmt.Printf("'world' ID: %d\n", id)
	// Output: 'world' ID: 1
}

// ExampleBPE_Type demonstrates getting the tokenizer type.
func ExampleBPE_Type() {
	tokens := []string{"hello"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	fmt.Printf("Type: %s\n", b.Type())
	// Output: Type: bpe
}

// ExampleBPE_BOSID demonstrates getting the BOS token ID.
func ExampleBPE_BOSID() {
	tokens := []string{"hello"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens, BOSID: 100, HasBOSID: true}

	b, _ := bpe.New(data)
	fmt.Printf("BOS ID: %d\n", b.BOSID())
	// Output: BOS ID: 100
}

// ExampleBPE_EOSID demonstrates getting the EOS token ID.
func ExampleBPE_EOSID() {
	tokens := []string{"hello"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens, EOSID: 200, HasEOSID: true}

	b, _ := bpe.New(data)
	fmt.Printf("EOS ID: %d\n", b.EOSID())
	// Output: EOS ID: 200
}

// ExampleBPE_SetCache demonstrates enabling encoding cache.
func ExampleBPE_SetCache() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	b.SetCache(100) // Cache up to 100 encoded results
	fmt.Println("Cache enabled with capacity 100")
	// Output: Cache enabled with capacity 100
}

// ExampleBPE_ClearCache demonstrates clearing the encoding cache.
func ExampleBPE_ClearCache() {
	tokens := []string{"hello", "world"}
	data := &common.TokenizerData{Model: "bpe", Tokens: tokens}

	b, _ := bpe.New(data)
	b.SetCache(100)
	b.ClearCache()
	fmt.Println("Cache cleared")
	// Output: Cache cleared
}
