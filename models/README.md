# Tokenization Package

Native GGUF tokenization and detokenization support for go-tokenizer. Reads vocabulary, merge rules, special token IDs, and pre-tokenization mode directly from GGUF files using `github.com/cymertek/go-gguf`.

## Quick Start

```go
import (
    "github.com/cymertek/go-gguf"
    "github.com/cymertek/go-tokenizer/models/common"
)

// Read tokenizer from GGUF file using go-gguf library
data, err := gguf.ReadTokenizerFromGGUF("model.gguf")
if err != nil {
    log.Fatal(err)
}

// Create appropriate tokenizer based on model type detected in KV data
tokenizer, err := common.New(data)
if err != nil {
    log.Fatal(err)
}

// Encode text to token IDs
ids := tokenizer.EncodeIDs("Hello world")
fmt.Println(ids) // []int{15043, 29871, ...}

// Decode back to text
text := tokenizer.Detokenize(ids)
fmt.Println(text) // "Hello world"
```

## Architecture

```
models/
├── gguf_reader.go        # ReadTokenizerFromGGUF — parses GGUF v3 KV section with tensor metadata skipping
├── gguf_types.go         # BType enum (Uint8-Float64, String, Array)
├── comprehensive_test.go # Integration tests with real GGUF files
├── all_models_test.go    # Cross-model comparison against llama.cpp reference outputs
├── common/               # Shared types and registry
│   ├── types.go          # TokenizerData struct with Has* flags for zero-value safety
│   ├── pretype.go        # PreType enum mapping GGUF tokenizer.ggml.pre strings
│   └── registry.go       # Constructor registration for all four formats
├── bpe/                  # BPE tokenizer sub-package (self-contained)
│   ├── bpe.go           # Trie-optimized BPE with concurrent encoding (>32KB inputs)
│   ├── trie.go          # Byte-level trie for O(k) longest-match vocabulary lookup
│   └── pretype.go       # Pre-tokenization strategies: GPT2, Llama3, Qwen2, StarCoder, etc.
├── spm/                  # SentencePiece tokenizer sub-package (self-contained)
│   ├── spm.go           # Greedy longest-match with ▁ space markers
│   ├── protobuf.go      # Minimal protobuf decoder for SPMModel binary
│   └── trie.go          # Byte-level trie
├── unigram/              # Unigram tokenizer sub-package (self-contained)
│   ├── unigram.go       # Viterbi-style DP with beam search using SPMProbabilities
│   └── trie.go          # Byte-level trie
└── wordpiece/            # WordPiece tokenizer sub-package (self-contained)
    ├── wordpiece.go     # BERT-style with ## continuation prefix
    └── trie.go          # Byte-level trie
```

## Key Types

### TokenizerData

Holds all tokenizer metadata extracted from a GGUF file. When constructed programmatically (not via `ReadTokenizerFromGGUF`), use `Has*` flags to distinguish "not set" from "set to zero".

```go
type TokenizerData struct {
    Model          string           // "bpe", "spm", "unigram", "wordpiece" — auto-detected from GGUF KV
    Tokens         []string         // Vocabulary tokens (position = token ID)
    Merges         []Merge          // BPE merge rules as pairs of string fragments
    SPMModel       []byte           // Protobuf-encoded SentencePiece model binary (from GGUF)
    BOSID, EOSID   int64            // default -1
    EOTID, EOMID   int64
    UNKID, PADID   int64
    HasBOSID, HasEOSID bool         // distinguishes unset (-1) from "set to zero"
    HasEOTID, HasEOMID bool
    HasUNKID, HasPADID bool
    AddBOS, AddEOS  bool             // prepend/append BOS/EOS tokens in EncodeIDs output
    PreType         common.PreType   // pre-tokenization strategy (BPE only)
    TokenType       []int32          // per-token type flags from GGUF (optional)
    SpaceChar       rune             // space prefix character detected from vocabulary (e.g., Ġ)
    SPMProbabilities []float64       // interleaved [id, score] pairs for Unigram Viterbi decoding
    Config          map[string]interface{}  // additional tokenizer-specific configuration
}
```

### BPE Tokenizer

Trie-optimized BPE tokenizer with greedy longest-match, pre-tokenization strategies, and concurrent encoding for large inputs.

```go
type BPE struct {
    vocab     map[string]int       // token string → id (built from Tokens)
    invVocab  map[int]string       // id → token string (for Detokenize)
    trie      *byteTrie            // O(k) trie-based matching for EncodeIDs
    bosID, eosID int64             // special token IDs (-1 if not configured)
    unkID     int64                // unknown token ID (-1 if not configured)
    preType   common.PreType       // pre-tokenization strategy (GPT2, Llama3, Qwen2, etc.)
    addBOS, addEOS bool           // whether to prepend/append BOS/EOS in EncodeIDs output
    spaceChar rune                  // space prefix character (e.g., U+0120 for Ġ)
    cache     map[string][]int     // optional encoding cache (SetCache enables)
}
```

### PreType Enum

Identifies the pre-tokenization strategy. Maps from GGUF `tokenizer.ggml.pre` strings to implementation modes.

```go
const (
    PreDefault PreType = iota    // Default sequential rules: punct → letters → digits → 3-digit numbers
    PreGPT2                      // GPT-2 / Phi-2 / MPT / OLMo / Jais: contractions-first, one digit at a time
    PreQwen2                     // Qwen2 / Megrez: greedy digit runs
    PreLlama3                    // Llama 3 / Falcon3 / Pixtral: max 3 digits
    PreStarcoder                 // StarCoder / Command-R / Refact: sequential numbers → letters → punct
    PreDeepSeekLLM               // DeepSeek-LLM / DeepSeek-Coder: newlines → letters → punct → CJK → digits
    PreFalcon                    // Falcon: punct → letters → 3-digit numbers
    PreQwen35                    // Qwen3.5: diacritics first, then greedy numbers
    PreStableLM2                 // StableLM2 / Hunyuan: same as Qwen2
    PreGPT4O                     // GPT-4o / Llama4: max 3 digits
    PreGemma4                    // Gemma 4: (reserved)
)
```

## GGUF File Reading with go-gguf

The `gguf` package provides efficient streaming access to GGUF files without loading entire tensors into memory.

### Reading Tokenizer Data Only

Use `ReadTokenizerFromGGUF()` to extract only KV entries (skipping tensor data section):

```go
import "github.com/cymertek/go-gguf"

data, err := gguf.ReadTokenizerFromGGUF("model.gguf")
if err != nil {
    log.Fatal(err)
}

// Verify extracted metadata
fmt.Printf("Architecture: %s\n", data.Model)
fmt.Printf("Vocabulary size: %d tokens\n", len(data.Tokens))
fmt.Printf("Merge rules: %d (BPE only)\n", len(data.Merges))
if data.HasBOSID {
    fmt.Printf("BOS token ID: %d\n", data.BOSID)
}
```

### How KV Section Parsing Works

`ReadTokenizerFromGGUF()` reads the GGUF v3 format:

1. **Header** (24 bytes): magic "GGUF" + version uint32 + n_kv_count uint64 + alignment uint64
2. **KV Entries**: key_len(uint64) + name(key_len bytes) + btype(uint32) + value(payload)
3. **Tensor Metadata Skip**: For keys ending in `.weight` or `.bias`, scan forward past tensor metadata (n_dims, shape dimensions) until next valid KV entry

The parser handles all BType values (0-12) including String arrays with preserved length prefixes for downstream tokenization.

## Supported Tokenizer Formats

### BPE (Byte Pair Encoding)
**Models**: GPT-2, Llama 3, Qwen 2.5, StarCoder, DeepSeek  
**Required Fields**: `Tokens`, `Merges`, `PreType`, `SpaceChar`  
**Features**: Trie-based O(k) matching, concurrent encoding for >32KB inputs

### SentencePiece (SPM)
**Models**: Falcon, MPT, CodeLlama  
**Required Fields**: `Tokens`, `SPMModel` (protobuf binary), `SpaceChar`  
**Features**: Greedy longest-match with ▁ space markers

### Unigram
**Models**: Gemma, PaLM  
**Required Fields**: `Tokens`, `SPMProbabilities` (interleaved [id, score] float64 pairs)  
**Features**: Viterbi dynamic programming for optimal segmentation

### WordPiece
**Models**: BERT, RoBERTa, DeBERTa  
**Required Fields**: `Tokens` (with `##` continuation prefix markers), special token IDs  
**Features**: Greedy longest-match after whitespace pre-tokenization

## Tests

```bash
# Run all format round-trip tests
go test ./models/ -v -run "TestRoundTrip"

# Test with real GGUF files (requires GGUF_PATH env var or Bonsai-8B.gguf)
GGUF_PATH=./Bonsai-8B.gguf go test ./models/ -run TestRealGGUFFile -v

# Benchmark BPE encoding performance
go test ./models/bpe/ -bench=EncodeIDs -benchmem
```

## Performance Characteristics

| Operation | Target | Notes |
|-----------|--------|-------|
| `EncodeCount` | < 300 ns/op | Zero allocations for GPT-2 vocab (151K tokens) |
| `EncodeIDs` | < 7,000 ns/op | Trie-based O(k) matching per token |
| `Detokenize` | < 100 ns/op | Inverse vocabulary lookup |
| Trie construction | O(V) | V = total vocabulary bytes, one-time cost |

## Dependencies

- **github.com/cymertek/go-gguf** — GGUF v3 file parsing with lazy reader and efficient KV streaming
- **No regexp dependency** in the models package (uses byte-level trie for all matching)
- **No regexpset** in the models package (Aho-Corasick or trie-based matching only)

## License

MIT License — see [LICENSE](../LICENSE) file for full terms.
