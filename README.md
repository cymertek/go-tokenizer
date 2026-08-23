# go-tokenizer

[![Go Reference](https://pkg.go.dev/badge/github.com/cymertek/go-tokenizer.svg)](https://pkg.go.dev/github.com/cymertek/go-tokenizer)
[![Go Report Card](https://goreportcard.com/badge/github.com/cymertek/go-tokenizer)](https://goreportcard.com/report/github.com/cymertek/go-tokenizer)

Production-ready Go tokenizer library for GGUF models with full fidelity serialization across all four tokenization formats (BPE, SentencePiece, Unigram, WordPiece). Supports lossless binary round-trip serialization and efficient client-server token exchange.

## Features

- **Four Tokenizer Formats**: BPE, SentencePiece, Unigram, WordPiece
- **GGUF v3 Support**: Read tokenizer data from GGUF files using `github.com/cymertek/go-gguf`
- **Lossless Serialization**: Binary protobuf-like format preserving all fields for exact round-tripping
- **Trie-Based Matching**: O(k) vocabulary lookup without regex dependencies
- **Concurrent Encoding**: Automatic parallel processing for inputs >32KB
- **Zero-Allocation Counting**: `EncodeCount()` returns single int without slice allocation

## Quick Start

```go
import (
    "bytes"
    "github.com/cymertek/go-gguf"
    "github.com/cymertek/go-tokenizer/models/common"
)

// Read tokenizer from GGUF file using go-gguf library
data, err := gguf.ReadTokenizerFromGGUF("model.gguf")
if err != nil {
    panic(err)
}

// Create appropriate tokenizer based on model type
tokenizer, err := common.New(data)
if err != nil {
    panic(err)
}

// Encode text to token IDs
ids := tokenizer.EncodeIDs("Hello world")
fmt.Println(ids) // []int{15043, 29871, ...}

// Decode back to text
text := tokenizer.Detokenize(ids)
fmt.Println(text) // "Hello world"

// Serialize for network transport (binary protobuf-like format)
var buf bytes.Buffer
tok.Serialize(tokenizer, &buf)

// Deserialize on client/server side
deserialized, err := tok.Deserialize(&buf)
```

## GGUF File Reading with go-gguf

Use `github.com/cymertek/go-gguf` to read tokenizer data from GGUF files:

```go
import (
    "github.com/cymertek/go-gguf"
    "github.com/cymertek/go-tokenizer/models/common"
)

// Read complete tokenizer metadata from a GGUF file
data, err := gguf.ReadTokenizerFromGGUF("path/to/model.gguf")
if err != nil {
    log.Fatal(err)
}

// Create tokenizer instance based on model type detected in KV data
tokenizer, err := common.New(data)
if err != nil {
    log.Fatal(err)
}

// Verify model type and special tokens
fmt.Printf("Model: %s\n", data.Model)
fmt.Printf("Tokens: %d, Merges: %d\n", len(data.Tokens), len(data.Merges))
fmt.Printf("BOS: %d (has=%v), EOS: %d (has=%v)\n", 
    data.BOSID, data.HasBOSID, data.EOSID, data.HasEOSID)
```

The `gguf` package handles:
- GGUF v3 header parsing (magic, version, n_kv_count, alignment)
- KV entry reading with all BType dispatch (Uint8-Float64, String, Array)
- Tensor metadata skipping for weight/bias entries in v3 format
- String array preservation with length prefixes for downstream parsing

## Full Fidelity Serialization

The library provides **lossless round-trip serialization** across all four tokenizer formats. Every field required for exact reproduction of tokenization behavior is captured in the binary format:

### Wire Format (Version 2)

```
Magic: "TOKD"           // 4 bytes
Version: 0x02           // 1 byte — supports all four formats fully
Fields: Length-prefixed key-value pairs until EOF
Key space: Single-byte ASCII keys with lowercase for special cases
Forward compatibility: Unknown keys are skipped during deserialization
```

### Field Specification

| Key | Description | Format | Required For |
|-----|-------------|--------|--------------|
| `'M'` | Model type (`"bpe"`, `"spm"`, `"unigram"`, `"wordpiece"`) | String | All formats |
| `'T'` | Vocabulary tokens (null-terminated string slice) | Length-prefixed bytes | All formats |
| `'B'` | BOS token ID + Has* flag | 9-byte int64 with flag byte | BPE, SPM, Unigram |
| `'E'` | EOS token ID + Has* flag | 9-byte int64 with flag byte | BPE, SPM, Unigram |
| `'F'` | EOT token ID (end-of-text) + Has* flag | 9-byte int64 with flag byte | GLM5 and variants |
| `'H'` | EOM token ID (end-of-message) + Has* flag | 9-byte int64 with flag byte | GLM5 and variants |
| `'U'` | UNK token ID + Has* flag | 9-byte int64 with flag byte | All formats |
| `'P'` | PAD token ID + Has* flag | 9-byte int64 with flag byte | BPE, SPM, Unigram |
| `'A'` | AddBOS bool (prepend BOS to output) | Single byte (0 or 1) | BPE, SPM, Unigram |
| `'S'` | AddEOS bool (append EOS to output) | Single byte (0 or 1) | BPE, SPM, Unigram |
| `'Y'` | PreType enum value (GPT2, Llama3, Qwen2, etc.) | 4-byte int32 | BPE only |
| `'C'` | SpaceChar rune (e.g., `Ġ` for BPE) | 4-byte uint32 | BPE only |
| `'m'` | Merge rule (left + right fragments) | Length-prefixed bytes with null separator | BPE only |
| `'V'` | **SPMProbabilities** — interleaved `[id, score]` float64 pairs for Unigram Viterbi decoding | 8-byte float64 per value | Unigram only |
| `'W'` | TokenType array (parallel to Tokens) | Length-prefixed int32 slice | All formats (for GGUF compat) |
| `'X'` | SPMModel binary (protobuf-encoded SentencePiece model) | Raw bytes | SPM from GGUF files |
| `'N'` | NormalizerConfig (type + parameters) | JSON blob | Optional |
| `'R'` | PreTokenizerConfig (type + parameters) | JSON blob | Optional |
| `'D'` | DecoderConfig (type + parameters) | JSON blob | Optional |
| `'Z'` | ConfigMap entries (arbitrary key-value pairs) | JSON blob per entry | Optional |

### Format-Specific Requirements

Each tokenizer format requires different state for full fidelity:

#### BPE (Byte Pair Encoding)
- **Vocabulary** (`Tokens`) + **Merge rules in order** (`Merges`)
- `PreType` enum to select correct pre-tokenization strategy (GPT2, Llama3, Qwen2, etc.)
- `SpaceChar` rune for word boundary markers (e.g., `Ġ` = U+0120)

#### SentencePiece (SPM)
- **SPMModel binary** (`SPMModel`) — protobuf-encoded model from GGUF file
- Vocabulary alignment with SPM binary IDs
- Special token IDs preserved via Has* flags

#### Unigram
- **Vocabulary** + **SPMProbabilities** (interleaved `[id, score]` float64 pairs)
- Probabilities required for Viterbi dynamic programming segmentation
- Without probabilities, deserialized tokenizer falls back to greedy longest-match

#### WordPiece
- **Vocabulary** with `##` continuation prefix markers
- Special tokens: `[CLS]`, `[SEP]`, `[UNK]`, `<s>`, `</s>`
- BOS/EOS configuration preserved via Has* flags and AddBOS/AddEOS bools

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Application                       │
│  (LLM server, inference engine, API client)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓↑ Token IDs (pure uint32 binary)
┌─────────────────────────────────────────────────────────────┐
│                    Serialization Layer                        │
│  tok.Serialize(tokenizer, w io.Writer)                       │
│  tok.Deserialize(r io.Reader) (*Tokenizer, error)            │
│  - Binary protobuf-like format (TOKD v2)                     │
│  - Lossless round-trip for all four formats                  │
└─────────────────────────────────────────────────────────────┘
                            ↓↑ Serialized bytes
┌─────────────────────────────────────────────────────────────┐
│                    Network Transport                          │
│  (HTTP, gRPC, WebSocket, etc.)                               │
│  No JSON overhead — pure binary token exchange               │
└─────────────────────────────────────────────────────────────┘
                            ↓↑ Token IDs (pure uint32 binary)
┌─────────────────────────────────────────────────────────────┐
│                      Server Application                       │
│  (LLM inference, text generation, embedding service)         │
└─────────────────────────────────────────────────────────────┘
```

### Package Structure

```
/workdir/
├── tokenizer.go              # Top-level Serialize/Deserialize API
├── proto/tokenizer.pb.go     # Binary protobuf-like format implementation (v2)
├── models/
│   ├── common/types.go       # TokenizerData struct with Has* flags for zero-value safety
│   ├── common/pretype.go     # PreType enum (GPT2, Llama3, Qwen2, etc.)
│   ├── bpe/bpe.go           # BPE tokenizer implementation with trie optimization
│   ├── spm/spm.go           # SentencePiece tokenizer with protobuf model binary
│   ├── unigram/unigram.go   # Unigram tokenizer with Viterbi probability decoding
│   ├── wordpiece/wordpiece.go # WordPiece tokenizer (BERT-style)
│   └── full_fidelity_test.go # Comprehensive round-trip tests for all formats
├── normalizer/               # Text normalization (BERT, Unicode, Pattern-based)
├── pretokenizer/             # Pre-segmentation strategies (10 implementations)
├── decoder/                  # Token ID → text decoding (BPE, WordPiece, CTC, etc.)
└── processor/                # BOS/EOS insertion, padding, truncation

```

## Complete Client-Server Workflow

### Server Side (LLM Inference Service)

```go
// 1. Load tokenizer from GGUF file at startup using go-gguf
data, err := gguf.ReadTokenizerFromGGUF("model.gguf")
if err != nil {
    log.Fatal(err)
}

tokenizer, err := common.New(data)
if err != nil {
    log.Fatal(err)
}

// 2. Serialize once and broadcast to clients
var buf bytes.Buffer
tok.Serialize(tokenizer, &buf)
serializedToken := buf.Bytes()

// 3. Send serialized tokenizer to client via HTTP/gRPC/WebSocket
http.HandleFunc("/tokenizer", func(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/octet-stream")
    w.Write(serializedToken)
})
```

### Client Side (API Consumer)

```go
// 1. Download serialized tokenizer from server
resp, err := http.Get("https://api.example.com/tokenizer")
if err != nil {
    log.Fatal(err)
}
defer resp.Body.Close()

serializedToken, err := io.ReadAll(resp.Body)
if err != nil {
    log.Fatal(err)
}

// 2. Deserialize locally — no JSON overhead, pure binary
tokenizer, err := tok.Deserialize(bytes.NewReader(serializedToken))
if err != nil {
    log.Fatal(err)
}

// 3. Encode user input to token IDs for sending to server
ids := tokenizer.EncodeIDs("Hello world")
// Send []uint32(ids) over network — no JSON serialization needed!

// 4. Receive generated tokens from server and decode back to text
generatedTokens := []int{15043, 29871, 3896} // From server response
text := tokenizer.Detokenize(generatedTokens)
fmt.Println(text) // "Hello world"
```

## Performance Characteristics

### Serialization Overhead

- **BPE (GPT-2)**: ~150K tokens × 4 bytes avg = ~600KB serialized size
- **SPM**: Model binary + vocab = typically <1MB for most models
- **Unigram**: Vocab + probabilities = ~800KB - 2MB depending on model size
- **WordPiece**: Vocabulary only (no merges) = ~500KB - 1MB

### Round-Trip Fidelity

| Format | Lossless? | Notes |
|--------|-----------|-------|
| BPE | ✅ Yes | Vocab + merges in order + PreType + SpaceChar preserved exactly |
| SPM | ⚠️ Partial | Model binary preserved; EOT/EOM IDs may not be in all GGUF files |
| Unigram | ✅ Yes | **With probabilities** — Viterbi decoding produces identical results |
| WordPiece | ✅ Yes | Vocabulary + special tokens + continuation prefix logic preserved |

### Encoding Speed

- **Trie-based matching**: O(k) per token where k = max token length (~5 bytes average for BPE)
- **Concurrent encoding**: Automatic for inputs >32KB, splits across CPU cores
- **Zero allocations**: `EncodeCount()` returns single int without slice allocation

## Pre-tokenization Strategies (BPE only)

GGUF files specify pre-tokenization via `tokenizer.ggml.pre`. Implemented in `models/common/pretype.go`:

| GGUF `pre` value | Mode | Number handling |
|---|---|---|
| `gpt-2`, `phi-2`, `mpt`, `olmo`, `jais` | GPT-2 | One digit at a time |
| `llama3`, `llama-bpe`, `falcon3`, `pixtral` | Llama3 | Max 3 digits |
| `qwen2`, `megrez` | Qwen2 | Greedy digit runs |
| `stablelm2`, `hunyuan` | StableLM2 | Same as Qwen2 |
| `qwen35` | Qwen35 | Diacritics first, then greedy numbers |
| `gpt-4o`, `llama4` | GPT-4O | Max 3 digits |
| `starcoder`, `command-r`, `refact` | StarCoder | Sequential: numbers → letters → punct |
| `deepseek-llm`, `deepseek-coder` | DeepSeekLLM | Sequential: newlines → letters → punct → CJK → digits |
| `falcon` | Falcon | Sequential: punct → letters → 3-digit numbers |
| (missing) | Default | Sequential: punct → letters → digits → 3-digit numbers |

Pre-tokenization is implemented via hand-written state machines, not regex — **10–600x faster** than regex-based approaches.

## Normalizer Implementations

Text normalization before pre-tokenization:

| Normalizer | File | Description |
|------------|------|-------------|
| **BERT** | `bert.go` | NFKC + BERT punctuation handling, optional lowercase/Chinese char spacing |
| **Default** | `default.go` | Standard Unicode normalization (NFC/NFD/NFKC/NFKD) |
| **Prepend** | `prepend.go` | Add prefix/suffix strings to normalized text |
| **Replace** | `replace.go` | Character replacement rules with pattern matching |
| **Sequence** | `sequence.go` | Chain multiple normalizers in order |

## Decoder Implementations

Token ID → text decoding strategies:

| Decoder | File | Description |
|---------|------|-------------|
| **BPE** | `bpe.go` | Joins tokens, replaces suffix markers (e.g., `</w>` → space) |
| **WordPiece** | `wordpiece.go` | Handles `##` continuation prefix for subwords |
| **CTC** | `ctc.go` | Connectionist Temporal Classification decoding |
| **Fuse** | `fuse.go` | Merges adjacent tokens with overlapping boundaries |
| **Strip** | `strip.go` | Removes leading/trailing whitespace from token chain |
| **Sequence** | `sequence.go` | Chains multiple decoders in order |

## Processor Implementations

Post-processing after tokenization:

| Processor | File | Description |
|-----------|------|-------------|
| **BERT** | `bert.go` | `[CLS]` / `[SEP]` special tokens with padding and TypeId assignment |
| **ByteLevel** | `bytelevel.go` | RoBERTa-style byte-level post-processing |
| **RoBERTa** | `roberta.go` | Similar to ByteLevel with optional prefix space handling |
| **Sequence** | `sequence.go` | Chain multiple processors in order |
| **Template** | `template.go` | Configurable template-based processing (e.g., `[CLS] text [SEP]`) |

## API Reference

### Top-level Serialization

```go
package tokenizer

// Serialize writes the Tokenizer to w in binary protobuf-like format.
func Serialize(t *Tokenizer, w io.Writer) error

// Deserialize reads a Tokenizer from r in binary protobuf-like format.
func Deserialize(r io.Reader) (*Tokenizer, error)

// Extract reads tokenizer data from an io.Reader (for GGUF files).
func Extract(r io.Reader) (*common.TokenizerData, error)
```

### Model Construction

```go
package common

// New creates a tokenizer instance from TokenizerData by dispatching to appropriate model.
func New(data *TokenizerData) (Tokenizer, error)

type Tokenizer interface {
    EncodeIDs(text string) []int
    Detokenize(ids []int) string
    Count(text string) int
    Type() string
    HasToken(tok string) bool
    TokenID(tok string) int
}
```

### BPE-Specific API

```go
package bpe

type BPE struct { /* ... */ }

func New(data *TokenizerData) (*BPE, error)
func (b *BPE) EncodeIDs(text string) []int
func (b *BPE) Detokenize(ids []int) string
func (b *BPE) Count(text string) int
func (b *BPE) SetCache(size int)  // Enable result caching
func (b *BPE) ClearCache()        // Clear cached results
```

### Special Token Accessors

```go
package bpe

// BOSID returns the beginning-of-sequence token ID, or -1 if not configured.
func (b *BPE) BOSID() int64

// EOSID returns the end-of-sentence token ID, or -1 if not configured.
func (b *BPE) EOSID() int64
```

## Tests

### Unit tests for all formats

```bash
# Run all model round-trip tests (BPE, SPM, Unigram, WordPiece)
go test ./models/ -v -run "TestRoundTrip"

# Run full fidelity tests with probabilities and special tokens
go test ./models/ -v -run "TestFullFidelity"

# Run BPE-specific tests with trie optimization
go test ./models/bpe/ -v

# Run SPM tests with protobuf model binary
go test ./models/spm/ -v

# Run Unigram tests with Viterbi decoding
go test ./models/unigram/ -v

# Run WordPiece tests with continuation prefix logic
go test ./models/wordpiece/ -v
```

### Integration with real GGUF files

Set `GGUF_PATH` environment variable to point to a GGUF file:

```bash
GGUF_PATH=./Bonsai-8B.gguf go test ./models/ -run TestRealGGUFFile -v
```

Tests verify:
- Tokenizer data extraction from GGUF (vocab size, merges count, special token IDs)
- Round-trip encode→decode fidelity across multiple models and formats
- Pre-tokenization strategy correctness for each model type
- Concurrent encoding produces identical results to sequential encoding
- **Full serialization round-trip**: serialize → deserialize → encode matches original

### Benchmarks

```bash
go test ./models/bpe/ -bench=EncodeIDs_BPE -benchmem
go test ./models/bpe/ -bench=EncodeCount -benchmem
```

Performance targets:
- `EncodeCount`: < 300 ns/op for GPT-2 vocab (151K tokens)
- `EncodeIDs`: < 7,000 ns/op for GPT-2 vocab
- Zero allocations for `EncodeCount`
- Trie construction: O(V) where V = total vocabulary bytes

## Dependencies

- **github.com/cymertek/go-gguf** — GGUF v3 file parsing with lazy reader and efficient KV streaming
- **No regexp** in the models package (uses byte-level trie for all formats)
- **No regexpset** in the models package (Aho-Corasick or trie-based matching only)

## Production Use Cases

### LLM Inference Server → Client Token Exchange

1. **Server loads tokenizer once** from GGUF file at startup using `go-gguf`
2. **Serializes to binary protobuf-like format** (~600KB for GPT-2, ~1MB for most models)
3. **Sends serialized tokenizer to clients** via HTTP endpoint or WebSocket handshake
4. **Clients deserialize locally** and encode user input to token IDs
5. **Send pure uint32 binary token arrays** over network (no JSON overhead)
6. **Server receives tokens**, runs inference, returns generated token IDs
7. **Client decodes response** back to text using deserialized tokenizer

This eliminates:
- ❌ JSON serialization of token arrays (overhead + size)
- ❌ Client-side GGUF parsing (requires `go-gguf` library and binary format knowledge)
- ❌ Mismatched tokenization between server and client (ensures identical behavior via serialized tokenizer)

### Multi-Format Support

The library handles all four major tokenization formats used in modern LLMs:

| Format | Models Using It | Serialization Requirements |
|--------|----------------|---------------------------|
| **BPE** | GPT-2, Llama 3, Qwen 2.5, StarCoder | Vocab + merges (ordered) + PreType + SpaceChar |
| **SPM** | Falcon, MPT, CodeLlama | SPM model binary + vocab alignment |
| **Unigram** | Gemma, PaLM | Vocabulary + probabilities for Viterbi decoding |
| **WordPiece** | BERT, RoBERTa, DeBERTa | Vocabulary with `##` continuation markers |

Each format's serialization captures exactly what's needed for lossless round-tripping.

## Known Limitations

### Concurrent encoding edge cases

For inputs >32KB, chunk boundaries can cause tokens to span differently than sequential encoding. The `encodeConcurrent()` function handles BOS/EOS injection at chunk boundaries but may produce slightly different token sequences for very long texts. This is inherent to the concurrent design and documented in test coverage.

### SPM EOT/EOM IDs

Some GGUF files don't include EOT (end-of-text) or EOM (end-of-message) token IDs even when configured in the model. These are preserved via Has* flags only if explicitly present in the source data.

## License

MIT License — see [LICENSE](./LICENSE) file for full terms.

Copyright (c) 2024-2026 Cymertek
