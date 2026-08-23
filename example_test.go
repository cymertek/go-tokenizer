package tokenizer_test

import (
	"bytes"
	"fmt"

	tok "github.com/cymertek/go-tokenizer"
	"github.com/cymertek/go-tokenizer/models/bpe"
	"github.com/cymertek/go-tokenizer/models/common"
)

// ExampleBPE_EncodeIDs demonstrates tokenizing with a GGUF BPE model.
func ExampleBPE_EncodeIDs() {
	data := &common.TokenizerData{
		Tokens: []string{"hello", "world", "Ġhello", "Ġworld"},
	}
	bpeTok, _ := bpe.New(data)

	ids := bpeTok.EncodeIDs("hello world")
	fmt.Println(ids)

	// Output: [0 3]
}

// ExampleSerialize demonstrates the round-trip of Serialize/Deserialize with FromSlice output.
func ExampleSerialize() {
	data := &common.TokenizerData{
		Model: "wordpiece",
		Tokens: []string{"[CLS]", "[SEP]", "hello", "world"},
		BOSID:  102,
		EOSID:  103,
		AddBOS: true,
		AddEOS: true,
	}
	data.HasBOSID = true
	data.HasEOSID = true

	tokenizer := tok.New(data)

	var buf bytes.Buffer
	if err := tok.Serialize(tokenizer, &buf); err != nil {
		fmt.Println("serialize error:", err)
		return
	}
	fmt.Printf("serialized %d bytes\n", buf.Len())

	recovered, err := tok.Deserialize(&buf)
	if err != nil {
		fmt.Println("deserialize error:", err)
		return
	}
	fmt.Println("model:", recovered.Data.Model)
	fmt.Println("tokens:", recovered.Data.Tokens)
	fmt.Println("bos:", recovered.Data.BOSID, "has:", recovered.Data.HasBOSID)
	fmt.Println("add_bos:", recovered.Data.AddBOS, "add_eos:", recovered.Data.AddEOS)

	// Output: serialized 106 bytes
	// model: wordpiece
	// tokens: [[CLS] [SEP] hello world]
	// bos: 102 has: true
	// add_bos: true add_eos: true
}
