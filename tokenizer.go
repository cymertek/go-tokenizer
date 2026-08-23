// Package tokenizer represents a tokenization pipeline.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"

	"github.com/cymertek/go-tokenizer/models/common"
	"github.com/cymertek/go-tokenizer/normalizer"
	"github.com/cymertek/go-tokenizer/proto"
)

// Token represents a single token with its ID, value string, and byte offsets.
type Token struct {
	// Id is the integer token identifier in the vocabulary. (Note: follows HuggingFace convention of using "Id" not "ID".)
	Id int
	// Value is the decoded text for this token (may include space markers).
	Value string
	// Offsets are [start, end] byte positions in the original input string.
	Offsets []int
}

// NewToken creates a new Token with the given id, value, and offsets.
func NewToken(id int, value string, offsets []int) Token {
	return Token{Id: id, Value: value, Offsets: offsets}
}

// PreTokenizer is in charge of doing the pre-segmentation step. It splits the given string
// in multiple substrings, keeping track of the offsets of said substrings from the
// `NormalizedString`. In some occasions, the `PreTokenizer` might need to modify the given
// `NormalizedString` to ensure we can entirely keep track of the offsets and the mapping with
// the original string.
type PreTokenizer interface {
	// PreTokenize splits a normalized string into pre-tokenized segments.
	PreTokenize(*PreTokenizedString) (*PreTokenizedString, error)
}

// Decoder is in charge of decoding tokens back into strings.
type Decoder interface {
	// Decode joins token strings into a single decoded text output.
	Decode(tokens []string) string
	// DecodeChain returns individual decoded segments for each input token.
	DecodeChain(tokens []string) []string
}

// PrepareEncodings prepares multiple encoding sequences for merging by filtering nil values.
func PrepareEncodings(encs ...*Encoding) []Encoding {
	var result []Encoding
	for _, e := range encs {
		if e != nil {
			result = append(result, *e)
		}
	}
	return result
}

// MergeEncodings merges multiple encoded sequences into a single encoding by concatenating all fields.
func MergeEncodings(encs []Encoding, _ ...bool) *Encoding {
	if len(encs) == 0 {
		return &Encoding{}
	}

	// Pre-pass: ensure every encoding has complete SequenceRanges derived from its TypeIds
	// (excluding special token positions). This handles cases where encodings were created
	// with TypeIds but without corresponding ranges. We use pointer dereference to modify
	// each slice element in place, since indexing returns a copy for non-pointer types.
	for i := range encs {
		e := &encs[i]
		if len(e.SequenceRanges) == 0 && len(e.TypeIds) > 0 {
			e.BuildRangesFromTypeIds(0, false)
		}
		// If still no ranges but TypeIds exist (e.g., CLS with empty SeqRanges),
		// normalize from TypeIds — this creates range entries for non-special tokens.
		if len(e.SequenceRanges) == 0 && len(e.TypeIds) > 0 {
			e.normalizeEmptySeqRanges()
		}
	}

	result := encs[0]

	// First pass: merge all fields except TypeIds and SequenceRanges
	for _, e := range encs[1:] {
		result.Ids = append(result.Ids, e.Ids...)
		result.Tokens = append(result.Tokens, e.Tokens...)
		result.Offsets = append(result.Offsets, e.Offsets...)
		if e.SpecialTokenMask != nil {
			result.SpecialTokenMask = append(result.SpecialTokenMask, e.SpecialTokenMask...)
		}
		if e.AttentionMask != nil {
			result.AttentionMask = append(result.AttentionMask, e.AttentionMask...)
		}
		if e.Words != nil {
			result.Words = append(result.Words, e.Words...)
		}
	}

	// Collect all sequence ranges with proper offset for encs[1+].
	var allRanges []RangeEntry

	for seqID, rng := range result.SequenceRanges {
		if len(rng) > 0 {
			allRanges = append(allRanges, RangeEntry{SeqID: seqID, Start: rng[0], End: rng[len(rng)-1]})
		}
	}

	// Track cumulative offset of previously merged encodings' lengths.
	cumulativeOffset := len(encs[0].Ids)
	for _, e := range encs[1:] {
		offset := cumulativeOffset
		for seqID, rng := range e.SequenceRanges {
			if len(rng) > 0 {
				allRanges = append(allRanges, RangeEntry{SeqID: seqID, Start: rng[0] + offset, End: rng[len(rng)-1] + offset})
			}
		}
		cumulativeOffset += len(e.Ids)
	}

	// Sort ranges by start position.
	sort.Slice(allRanges, func(i, j int) bool {
		return allRanges[i].Start < allRanges[j].Start
	})

	// Collect positions covered by special tokens so we can exclude them from ranges.
	specialPositions := make(map[int]bool)
	if result.SpecialTokenMask != nil {
		for i, m := range result.SpecialTokenMask {
			if m == 1 {
				specialPositions[i] = true
			}
		}
	}

	// Filter out ranges that are entirely special-token positions (e.g., CLS/SEP single tokens).
	var filteredRanges []RangeEntry
	for _, r := range allRanges {
		allSpecial := true
		for pos := r.Start; pos <= r.End; pos++ {
			if !specialPositions[pos] {
				allSpecial = false
				break
			}
		}
		if !allSpecial {
			filteredRanges = append(filteredRanges, r)
		}
	}

	// Rebuild SequenceRanges map from filtered entries.
	result.SequenceRanges = make(map[int]Range)
	for _, r := range filteredRanges {
		rng := make(Range, r.End-r.Start+1)
		for i := range rng {
			rng[i] = r.Start + i
		}
		result.SequenceRanges[r.SeqID] = rng
	}

	// Build TypeIds: start with defaults (0), overlay ranges, then fill uncovered positions
	// from the original encodings' TypeIds. This handles special-only tokens like CLS/SEP that
	// were filtered out of ranges but still need their correct TypeId.
	typeIDs := make([]int, len(result.Ids)) // default 0
	for _, r := range filteredRanges {
		for pos := r.Start; pos <= r.End && pos < len(typeIDs); pos++ {
			typeIDs[pos] = r.SeqID
		}
	}
	// Fill gaps from original encodings' TypeIds (offset by cumulative lengths).
	gapOffset := 0
	for _, e := range encs {
		for i, tid := range e.TypeIds {
			pos := gapOffset + i
			if pos < len(typeIDs) && typeIDs[pos] == 0 {
				typeIDs[pos] = tid
			}
		}
		gapOffset += len(e.Ids)
	}
	result.TypeIds = typeIDs
	return &result
}

// Model represents a model used during tokenization (i.e., BPE, WordPiece, or Unigram).
type Model interface {
	// Tokenize splits the given sequence into underlying tokens with byte offsets relative to the input.
	Tokenize(sequence string) ([]Token, error)
	// TokenToID finds the integer ID associated with a token string.
	TokenToID(token string) (int, bool)
	// IDToToken finds the string token associated with an integer ID.
	IDToToken(id int) (string, bool)
	// GetVocabSize returns the size of the vocabulary.
	GetVocabSize() int
}

// OffsetMapping is a marker interface for offset mapping implementations used during decoding.
type OffsetMapping interface {
	OffsetMapping()
}

// PostProcessor handles post-processing of encoded sequences (e.g., adding special tokens).
type PostProcessor interface {
	AddedTokens(isPair bool) int
	Process(encoding, pairEncoding *Encoding, addSpecialTokens bool) *Encoding
}

// DefaultProcess returns the first encoding unchanged when addSpecialTokens is false.
func DefaultProcess(encoding, pairEncoding *Encoding, addSpecialTokens bool) *Encoding {
	if !addSpecialTokens {
		return encoding
	}
	// When adding special tokens and a pair exists, merge them
	if pairEncoding != nil && encoding != nil {
		return MergeEncodings([]Encoding{*encoding, *pairEncoding}, false)
	}
	return encoding
}

// Tokenizer is the complete, serializable tokenizer with all components.
type Tokenizer struct {
	Data       *common.TokenizerData
	Normalizer normalizer.Normalizer
	PreTok     PreTokenizer
}

// New creates a Tokenizer from data with optional normalizer and pre-tokenizer.
func New(data *common.TokenizerData, opts ...TokenizerOption) *Tokenizer {
	t := &Tokenizer{Data: data}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// TokenizerOption is a function that configures a Tokenizer.
type TokenizerOption func(*Tokenizer)

// WithNormalizer sets the normalizer on the tokenizer.
func WithNormalizer(n normalizer.Normalizer) TokenizerOption {
	return func(t *Tokenizer) {
		t.Normalizer = n
	}
}

// WithPreTok sets the pre-tokenizer on the tokenizer.
func WithPreTok(p PreTokenizer) TokenizerOption {
	return func(t *Tokenizer) {
		t.PreTok = p
	}
}

// Clone returns a deep copy of the Tokenizer. Safe to modify without affecting original.
func (t *Tokenizer) Clone() *Tokenizer {
	if t == nil || t.Data == nil {
		return nil
	}

	// Deep copy TokenizerData fields for safe modification without affecting original.
	dataCopy := &common.TokenizerData{
		Model:  t.Data.Model,
		Tokens: append([]string(nil), t.Data.Tokens...),
		Merges: append([]common.Merge{}, t.Data.Merges...),
		BOSID:  t.Data.BOSID,
		EOSID:  t.Data.EOSID,
		AddBOS: t.Data.AddBOS,
		AddEOS: t.Data.AddEOS,
	}
	dataCopy.HasBOSID = t.Data.HasBOSID
	dataCopy.HasEOSID = t.Data.HasEOSID

	result := &Tokenizer{Data: dataCopy}

	norm := t.Normalizer
	if norm != nil {
		if cloner, ok := norm.(interface{ Clone() normalizer.Normalizer }); ok {
			result.Normalizer = cloner.Clone()
			return result
		}
		result.Normalizer = norm // share non-clonable normalizers (reference semantics)
	}

	preTok := t.PreTok
	if preTok != nil {
		if cloner, ok := preTok.(interface{ Clone() PreTokenizer }); ok {
			result.PreTok = cloner.Clone()
			return result
		}
		result.PreTok = preTok // share non-clonable pre-tokenizers (reference semantics)
	}

	return result
}

// Append adds tokens to the end of the vocabulary. Returns error if duplicates found.
func (t *Tokenizer) Append(tokens ...string) error {
	if t == nil || t.Data == nil {
		return fmt.Errorf("Append: tokenizer or data is nil")
	}

	tokenSet := make(map[string]bool, len(t.Data.Tokens))
	for _, tok := range t.Data.Tokens {
		tokenSet[tok] = true
	}

	for _, newTok := range tokens {
		if tokenSet[newTok] {
			return fmt.Errorf("Append: token %q already exists", newTok)
		}
		t.Data.Tokens = append(t.Data.Tokens, newTok)
		tokenSet[newTok] = true
	}

	return nil
}

// Union returns a new Tokenizer with deduplicated tokens from both this and other.
func (t *Tokenizer) Union(other *Tokenizer) (*Tokenizer, error) {
	if t == nil || other == nil || t.Data == nil || other.Data == nil {
		return nil, fmt.Errorf("Union: tokenizer or data is nil")
	}

	union := t.Clone()
	tokenSet := make(map[string]bool, len(union.Data.Tokens))
	for _, tok := range union.Data.Tokens {
		tokenSet[tok] = true
	}

	for _, tok := range other.Data.Tokens {
		if !tokenSet[tok] {
			union.Data.Tokens = append(union.Data.Tokens, tok)
			tokenSet[tok] = true
		}
	}

	return union, nil
}

// Intersect returns a new Tokenizer with only tokens that exist in both this and other.
func (t *Tokenizer) Intersect(other *Tokenizer) (*Tokenizer, error) {
	if t == nil || other == nil || t.Data == nil || other.Data == nil {
		return nil, fmt.Errorf("Intersect: tokenizer or data is nil")
	}

	otherSet := make(map[string]bool, len(other.Data.Tokens))
	for _, tok := range other.Data.Tokens {
		otherSet[tok] = true
	}

	intersection := &Tokenizer{Data: &common.TokenizerData{Model: t.Data.Model}}
	for _, tok := range t.Data.Tokens {
		if otherSet[tok] {
			intersection.Data.Tokens = append(intersection.Data.Tokens, tok)
		}
	}

	return intersection, nil
}

// Trim returns a new Tokenizer with tokens from other removed.
func (t *Tokenizer) Trim(other *Tokenizer) (*Tokenizer, error) {
	if t == nil || other == nil || t.Data == nil || other.Data == nil {
		return nil, fmt.Errorf("Trim: tokenizer or data is nil")
	}

	otherSet := make(map[string]bool, len(other.Data.Tokens))
	for _, tok := range other.Data.Tokens {
		otherSet[tok] = true
	}

	result := t.Clone()
	var kept []string
	for _, tok := range result.Data.Tokens {
		if !otherSet[tok] {
			kept = append(kept, tok)
		}
	}
	result.Data.Tokens = kept

	return result, nil
}

// Serialize writes the Tokenizer to w in binary protobuf format (.td).
func Serialize(t *Tokenizer, w io.Writer) error {
	if t == nil || t.Data == nil {
		return fmt.Errorf("Serialize: tokenizer or data is nil")
	}

	pt := &proto.Tokenizer{
		Data: t.Data,
	}

	// Serialize normalizer config if present — capture type and parameters for reconstruction.
	if t.Normalizer != nil {
		pt.Normalizer = serializeNormalizer(t.Normalizer)
	}

	// Serialize pre-tokenizer config if present.
	if t.PreTok != nil {
		pt.PreTok = serializePreTokenizer(t.PreTok)
	}

	return pt.Serialize(w)
}

func serializeNormalizer(n normalizer.Normalizer) *proto.NormalizerConfig {
	switch norm := n.(type) {
	case *normalizer.BertNormalizer:
		cfg, _ := json.Marshal(norm)
		return &proto.NormalizerConfig{Type: "bert", Config: cfg}
	default:
		return &proto.NormalizerConfig{Type: fmt.Sprintf("%T", n)}
	}
}

func serializePreTokenizer(p PreTokenizer) *proto.PreTokenizerConfig {
	if p == nil {
		return nil
	}
	// Store type name as string for reconstruction by caller.
	typeName := fmt.Sprintf("%T", p)
	switch typeName {
	case "*pretokenizer.Whitespace":
		return &proto.PreTokenizerConfig{Type: "whitespace"}
	case "*pretokenizer.WhitespaceSplit":
		return &proto.PreTokenizerConfig{Type: "whitespace_split"}
	default:
		return &proto.PreTokenizerConfig{Type: typeName}
	}
}

// Deserialize reads a Tokenizer from r in binary protobuf format (.td).
// Note: Normalizer and PreTokenizer are serialized as type-name strings only.
// Users must reconstruct these manually after deserialization if needed for full fidelity.
func Deserialize(r io.Reader) (*Tokenizer, error) {
	pt, err := proto.Deserialize(r)
	if err != nil {
		return nil, fmt.Errorf("Deserialize: %w", err)
	}

	t := &Tokenizer{Data: pt.Data}

	// Note: Normalizer and PreTok configs are stored as type-name strings only.
	// For full fidelity reconstruction, users should configure these manually:
	//   t.Normalizer = normalizer.NewBertNormalizer(...)
	//   t.PreTok = pretokenizer.NewWhitespace()

	return t, nil
}

// Extract reads tokenizer data from an io.Reader. For GGUF files, users should first open
// the file with their preferred method and pass the reader here.
func Extract(r io.Reader) (*common.TokenizerData, error) {
	// This is a placeholder — actual implementation depends on gguf package API
	return nil, fmt.Errorf("Extract: not yet implemented (requires gguf package integration)")
}
