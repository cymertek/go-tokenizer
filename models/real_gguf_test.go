package models

import (
	"os"
	"path/filepath"
	"testing"
)

// TestRealGGUFFile exercises tokenization with a real GGUF model file (GPT-2).
func TestRealGGUFFile(t *testing.T) {
	ggufPath := os.Getenv("GGUF_PATH")
	if ggufPath == "" {
		// Fall back to the test.gguf in the repo.
		ggufPath = filepath.Join("..", "..", "test.gguf")
		if _, err := os.Stat(ggufPath); err != nil {
			t.Skip("Set GGUF_PATH or provide test.gguf to test ReadTokenizerFromGGUF with a real file")
		}
	}

	data, err := ReadTokenizerFromGGUF(ggufPath)
	if err != nil {
		t.Fatalf("ReadTokenizerFromGGUF(%q): %v", ggufPath, err)
	}

	if data.Model != "gpt2" {
		t.Errorf("Model = %q, want %q", data.Model, "gpt2")
	}
	if !data.HasBOSID || data.BOSID != 151643 {
		t.Errorf("BOSID = %d, want 151643", data.BOSID)
	}
	if !data.HasEOSID || data.EOSID != 151645 {
		t.Errorf("EOSID = %d, want 151645", data.EOSID)
	}
	if !data.HasPADID || data.PADID != 151643 {
		t.Errorf("PADID = %d, want 151643", data.PADID)
	}
	if len(data.Tokens) == 0 {
		t.Error("Tokens should not be empty")
	}
	if len(data.Merges) == 0 {
		t.Error("Merges should not be empty")
	}

	bpe, err := NewBPE(data)
	if err != nil || bpe == nil {
		t.Fatalf("NewBPE error: %v", err)
	}
	if bpe.Count("Hello world") == 0 {
		t.Error("Count('Hello world') should be > 0")
	}

	// "Hello world" is a common GPT-2 test string.
	ids := bpe.EncodeIDs("Hello world")
	if len(ids) == 0 {
		t.Error("EncodeIDs('Hello world') should return tokens")
	} else {
		t.Logf("EncodeIDs('Hello world') = %v (len=%d)", ids, len(ids))
	}

	// "test" should tokenize to a small number of tokens.
	ids = bpe.EncodeIDs("test")
	if len(ids) == 0 {
		t.Error("EncodeIDs('test') should return tokens")
	} else {
		t.Logf("EncodeIDs('test') = %v (len=%d)", ids, len(ids))
	}

	// Count should match.
	hwIds := bpe.EncodeIDs("Hello world")
	count := bpe.Count("Hello world")
	if count != len(hwIds) {
		t.Logf("Count('Hello world') = %d vs len(EncodeIDs) = %d", count, len(hwIds))
	}
	count2 := bpe.Count("test")
	ids2 := bpe.EncodeIDs("test")
	if count2 != len(ids2) {
		t.Logf("Count('test') = %d vs len(EncodeIDs) = %d", count2, len(ids2))
	}

	// HasToken for a known token.
	// "<unk>" is typically at index 0 in GPT-2 vocab.
	if len(data.Tokens) > 0 {
		if !bpe.HasToken(data.Tokens[0]) {
			t.Errorf("HasToken(%q) should be true", data.Tokens[0])
		}
	}

	// TokenID for a known token.
	if len(data.Tokens) > 0 {
		id := bpe.TokenID(data.Tokens[0])
		if id != 0 {
			t.Errorf("TokenID(%q) = %d, want 0", data.Tokens[0], id)
		}
	}

	// Tokens should return the actual token strings.
	toks := bpe.Tokens("test")
	if len(toks) > 0 {
		t.Logf("Tokens('test') = %v", toks)
	}
}

func TestRealGGUFFile_NilReceiver(t *testing.T) {
	var bpe *BPE

	if bpe.HasToken("hello") {
		t.Error("nil BPE.HasToken should return false")
	}
	if bpe.TokenID("hello") != -1 {
		t.Error("nil BPE.TokenID should return -1")
	}
	if bpe.Count("hello") != 0 {
		t.Error("nil BPE.Count should be 0")
	}
	if bpe.EncodeIDs("hello") != nil {
		t.Error("nil BPE.EncodeIDs should return nil")
	}
	if bpe.Tokens("hello") != nil {
		t.Error("nil BPE.Tokens should return nil")
	}
}

func TestReadTokenizerFromGGUF_NilPath(t *testing.T) {
	_, err := ReadTokenizerFromGGUF("")
	if err == nil {
		t.Error("ReadTokenizerFromGGUF(\"\") should return an error")
	}
}
