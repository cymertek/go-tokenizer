package tokenizer

import (
	"fmt"

	"github.com/cymertek/go-tokenizer/util"
)

// PaddingDirection specifies whether padding should be applied to the left or right of an encoding.
type PaddingDirection int

const (
	// Left indicates padding should be prepended to encodings.
	Left PaddingDirection = iota
	// Right indicates padding should be appended to encodings.
	Right
)

// Encoding represents the output of a tokenizer with all tokenization metadata.
type Encoding struct {
	// Ids contains the integer token IDs produced by the tokenizer.
	Ids []int
	// TypeIds assigns a type ID to each token for multi-sequence handling (e.g., [CLS] vs [SEP]).
	TypeIds []int
	// Tokens holds the decoded string representation of each token ID.
	Tokens []string
	// Offsets contains byte position ranges [start, end] mapping tokens back to original input.
	Offsets [][]int
	// SpecialTokenMask flags positions that are special tokens (e.g., [CLS], [SEP]) with 1s.
	SpecialTokenMask []int
	// AttentionMask identifies padding tokens for the attention mechanism with 0s.
	AttentionMask []int
	// Overflowing contains a list of overflow encodings generated when being truncated.
	Overflowing []Encoding
	// Words holds optional indexes mapping each token to its source word (-1 if no mapping).
	Words []int
	// SequenceRanges maps sequence IDs to ranges of tokens covered by that sequence.
	SequenceRanges map[int]Range
}

// EncodingOpts configures encoding construction options like words and sequence ranges.
type EncodingOpts struct {
	// Words is an optional slice of word indexes for each token.
	Words []int
	// SequenceRange maps sequence IDs to their token ranges.
	SequenceRange map[int]Range
}

// RangeEntry represents a sequence range with its start/end positions and associated sequence ID.
type RangeEntry struct {
	SeqID int
	Start int
	End   int
}

// EncodingOpt is a function that configures encoding construction options.
type EncodingOpt func(o *EncodingOpts)

// WithWordsEncodingOpt sets the Words field for encoding construction.
func WithWordsEncodingOpt(v []int) EncodingOpt {
	return func(o *EncodingOpts) {
		o.Words = v
	}
}

// WithSequenceRangeEncodingOpt sets the SequenceRange map for encoding construction.
func WithSequenceRangeEncodingOpt(v map[int]Range) EncodingOpt {
	return func(o *EncodingOpts) {
		o.SequenceRange = v
	}
}

// DefaultEncodingOpts returns default encoding options with nil Words and empty Range map.
func DefaultEncodingOpts() *EncodingOpts {
	return &EncodingOpts{
		Words:         nil,
		SequenceRange: make(map[int]Range),
	}
}

// NewEncoding creates a new Encoding from the given token data.
func NewEncoding(ids []int, typeIDs []int, tokens []string, offsets [][]int, specialTokenMask []int, attentionMask []int, overflowing []Encoding, opts ...EncodingOpt) *Encoding {
	o := DefaultEncodingOpts()
	for _, opt := range opts {
		opt(o)
	}

	return &Encoding{
		ids,
		typeIDs,
		tokens,
		offsets,
		specialTokenMask,
		attentionMask,
		overflowing,
		o.Words,
		o.SequenceRange,
	}
}

// NewEncodingWithCapacity creates an Encoding pre-allocated with the given capacity for all fields.
func NewEncodingWithCapacity(l int) (retVal *Encoding) {
	return &Encoding{
		Ids:              make([]int, l),
		TypeIds:          make([]int, l),
		Tokens:           make([]string, l),
		Offsets:          make([][]int, l),
		SpecialTokenMask: make([]int, l),
		AttentionMask:    make([]int, l),
		Words:            make([]int, l),
		SequenceRanges:   make(map[int]Range),
	}
}

// DefaultEncoding returns an Encoding with empty slices and nil optional fields.
func DefaultEncoding() *Encoding {
	return &Encoding{
		Ids:              []int{},
		TypeIds:          []int{},
		Tokens:           []string{},
		Offsets:          [][]int{},
		SpecialTokenMask: []int{},
		AttentionMask:    []int{},
		Words:            nil,
		SequenceRanges:   make(map[int]Range),
	}
}

// NewEncodingFromTokens creates an Encoding from a slice of Token structs with the given type ID.
func NewEncodingFromTokens(tokens []Token, typeID int) (retVal *Encoding) {
	var (
		ids     []int
		offsets [][]int
		toks    []string
	)
	for _, t := range tokens {
		ids = append(ids, t.Id)
		offsets = append(offsets, t.Offsets)
		toks = append(toks, t.Value)
	}

	typeIDs := make([]int, len(tokens))
	// words := make([]int, len(tokens))
	var words []int
	specialTokenMask := util.Repeat(0, len(tokens))
	attentionMask := util.Repeat(1, len(tokens))

	return &Encoding{
		Ids:              ids,
		TypeIds:          typeIDs,
		Tokens:           toks,
		Offsets:          offsets,
		SpecialTokenMask: specialTokenMask,
		AttentionMask:    attentionMask,
		Overflowing:      []Encoding{},
		Words:            words,
		SequenceRanges:   make(map[int]Range),
	}
}

// Clone creates a deep copy of the Encoding using reflection-based serialization.
func (e *Encoding) Clone() *Encoding {
	out := new(Encoding)
	err := util.DeepCopy(e, out)
	if err != nil {
		panic(err)
	}

	return out
}

// IsEmpty returns whether Encoding is empty
func (e *Encoding) IsEmpty() (retVal bool) {
	return len(e.Ids) == 0
}

// Len returns number of encoding tokens
func (e *Encoding) Len() (retVal int) {
	return len(e.Ids)
}

// NSequences returns number of sequences combined in this encoding.
func (e *Encoding) NSequences() int {
	if len(e.SequenceRanges) == 0 {
		return 1
	}

	return len(e.SequenceRanges)
}

// GetTokens returns the token strings for this encoding.
func (e *Encoding) GetTokens() []string {
	return e.Tokens
}

// GetWords returns word indexes on normalized string
func (e *Encoding) GetWords() []int {
	return e.Words
}

// SetWord set word index value at given index of word in e.Words slice
func (e *Encoding) SetWord(index int, val int) {
	e.Words[index] = val
}

// SetSequenceIds set the given sequence id for the whole range of tokens contained in this Encoding.
// Only applies if SequenceRanges is not already populated. Special token positions (where
// SpecialTokenMask[i] == 1) are excluded from ranges — they retain their TypeId but don't belong to any sequence.
func (e *Encoding) SetSequenceIds(sequenceID int) {
	if e.Len() > 0 && len(e.SequenceRanges) == 0 {
		e.BuildRangesFromTypeIds(sequenceID, true)
	}
}

// BuildRangesFromTypeIds constructs SequenceRanges from TypeIds or a default sequence ID, excluding special token positions.
// If useDefault is false and TypeIds exist, uses those; otherwise defaults to the given sequenceID.
func (e *Encoding) BuildRangesFromTypeIds(defaultSequenceID int, useDefault bool) {
	if e.SequenceRanges == nil {
		e.SequenceRanges = make(map[int]Range)
	}

	type run struct {
		seqID int
		start int
		end   int // exclusive
	}
	var runs []run

	for i := 0; i < e.Len(); i++ {
		isSpecial := e.SpecialTokenMask != nil && len(e.SpecialTokenMask) > i && e.SpecialTokenMask[i] == 1
		if isSpecial {
			continue
		}
		seqID := defaultSequenceID
		if !useDefault && i < len(e.TypeIds) {
			seqID = e.TypeIds[i]
		}

		if len(runs) > 0 && runs[len(runs)-1].seqID == seqID && runs[len(runs)-1].end == i {
			runs[len(runs)-1].end++
		} else {
			runs = append(runs, run{seqID: seqID, start: i, end: i + 1})
		}
	}

	for _, r := range runs {
		existing := e.SequenceRanges[r.seqID]
		if len(existing) == 0 || existing[len(existing)-1]+1 < r.start {
			e.SequenceRanges[r.seqID] = NewRange(r.start, r.end)
		} else {
			newEntries := make(Range, r.end-r.start)
			for i := r.start; i < r.end; i++ {
				newEntries[i-r.start] = i
			}
			e.SequenceRanges[r.seqID] = append(existing, newEntries...)
		}
	}
}

// normalizeEmptySeqRanges assigns TypeIds to SequenceRanges for encodings that have
// TypeIds but no ranges. It groups consecutive non-special positions by their TypeId
// and creates range entries so they participate correctly during MergeEncodings.
func (e *Encoding) normalizeEmptySeqRanges() {
	type run struct {
		seqID int
		start int
		end   int
	}
	var runs []run

	for i := 0; i < e.Len(); i++ {
		isSpecial := e.SpecialTokenMask != nil && len(e.SpecialTokenMask) > i && e.SpecialTokenMask[i] == 1
		if isSpecial {
			continue
		}
		seqID := 0
		if i < len(e.TypeIds) {
			seqID = e.TypeIds[i]
		}

		if len(runs) > 0 && runs[len(runs)-1].seqID == seqID && runs[len(runs)-1].end == i {
			runs[len(runs)-1].end++
		} else {
			runs = append(runs, run{seqID: seqID, start: i, end: i + 1})
		}
	}

	if len(runs) > 0 {
		e.SequenceRanges = make(map[int]Range)
		for _, r := range runs {
			rng := make(Range, r.end-r.start)
			for j := r.start; j < r.end; j++ {
				rng[j-r.start] = j
			}
			e.SequenceRanges[r.seqID] = rng
		}
		return
	}

	// All tokens are special (e.g., a single CLS or SEP). Do NOT create ranges for these —
	// they should not appear in SequenceRanges. Their TypeIds will be applied during merge
	// gap-filling instead, so the merged result still gets correct TypeId values.
}


// GetSequenceIds returns a flat slice of sequence IDs corresponding to each token position.
func (e *Encoding) GetSequenceIds() []int {
	sequences := make([]int, e.Len())
	for seqID := 0; seqID < e.NSequences(); seqID++ {
		r := e.SequenceRanges[seqID]
		seqLen := r.Len()
		var a []int
		for range seqLen {
			a = append(a, seqID)
		}
		// replace items in Range r with seqID
		start := r[0]
		end := r[len(r)-1]
		before := sequences[:start]
		after := sequences[end:]
		sequences = append(before, a...)
		sequences = append(sequences, after...)
	}

	return sequences
}

// GetIds returns Ids from encoding
func (e *Encoding) GetIds() []int {
	return e.Ids
}

// GetTypeIds returns type Ids from encoding
func (e *Encoding) GetTypeIds() []int {
	return e.TypeIds
}

func (e *Encoding) SetTypeIds(typeIds []int) {
	e.TypeIds = typeIds
}

// GetOffsets returns offsets from encoding
func (e *Encoding) GetOffsets() [][]int {
	return e.Offsets
}

// GetSpecialTokenMask returns specialTokenMask from encoding
func (e *Encoding) GetSpecialTokenMask() []int {
	return e.SpecialTokenMask
}

// GetAttentionMask returns attentionMask from encoding
func (e *Encoding) GetAttentionMask() []int {
	return e.AttentionMask
}

// GetOverflowing returns overflowing from encoding
func (e *Encoding) GetOverflowing() []Encoding {
	return e.Overflowing
}

// SetOverflowing set overflowing.
func (e *Encoding) SetOverflowing(overflowing []Encoding) {
	e.Overflowing = overflowing
}

// TakeOverflowing returns overflowing and reset it to empty at encoding
func (e *Encoding) TakeOverflowing() []Encoding {
	o := e.Overflowing
	e.Overflowing = []Encoding{}
	return o
}

// Word2Tokens gets the encoded tokens corresponding the word
// at the given index in the input sequence
// in the form `(startToken, endToken + 1)`
//
// NOTE. e.Words is optional, therefore, there's case of `none` result
// if `none` result, `ok` will be false.
func (e *Encoding) Word2Tokens(word int) (startTok, endTok int, ok bool) {

	var inRangeWords []int
	for _, w := range e.Words {
		if w <= word {
			inRangeWords = append(inRangeWords, w)
		}
	}
	start := -1
	for i, w := range inRangeWords {
		if w == word && start < 0 {
			start = i
		}
	}

	end := len(inRangeWords)

	if start != -1 && end != -1 {
		return start, end, true
	} else {
		return startTok, endTok, false
	}
}

// Word2Chars get the offsets of the word at a given index in
// the input sequence
func (e *Encoding) Word2Chars(word int) (retVal []int, ok bool) {
	start, end, ok := e.Word2Tokens(word)
	if !ok || end == 0 {
		return retVal, false
	}
	oStart := e.Offsets[start][0]
	oEnd := e.Offsets[end-1][1]
	return []int{oStart, oEnd}, true
}

// Token2Chars get the offsets of the token at the given index
func (e *Encoding) Token2Chars(tokenIdx int) (retVal []int, ok bool) {
	if tokenIdx < 0 || tokenIdx > len(e.Offsets) {
		return retVal, false
	} else {
		return e.Offsets[tokenIdx], true
	}
}

// Token2Word returns the word index for a given token index.
// Returns false if tokenIdx is out of bounds or not assigned to any word.
func (e *Encoding) Token2Word(tokenIdx int) (retVal int, ok bool) {
	if tokenIdx < 0 || tokenIdx >= len(e.Words) {
		return -1, false
	}
	wordID := e.Words[tokenIdx]
	if wordID == -1 {
		return -1, false
	}
	return wordID, true
}

// Char2Token returns a token index that contains the given `char` index
func (e *Encoding) Char2Token(pos int) (retVal int, ok bool) {
	for i, o := range e.Offsets {
		if pos >= o[0] && pos < o[1] {
			return i, true
		}
	}

	return -1, false
}

// Char2Word get the word index that contain the given `char` index
func (e *Encoding) Char2Word(pos int) (retVal int, ok bool) {
	if idx, ok := e.Char2Token(pos); ok {
		return e.Token2Word(idx)
	}

	return -1, false
}

// Truncate truncates the current encoding
func (e *Encoding) Truncate(maxLen int, stride int) (retVal *Encoding, err error) {

	if stride >= maxLen || maxLen == 0 {
		return retVal, fmt.Errorf("invalid input maxLen or stride (stride must be less than maxLen and maxLen must be greater than zero.)")
	}

	if maxLen >= len(e.Ids) {
		// do nothing
		return e, nil
	}

	// Truncating at maxLen (exclusive) to keep.
	// The rest (overflowing) from maxLen (inclusive)
	newIds := e.Ids[0:maxLen]
	oIds := e.Ids[maxLen:len(e.Ids)] // overflowing
	newTypeIds := e.TypeIds[0:maxLen]
	oTypeIds := e.TypeIds[maxLen:len(e.TypeIds)]
	newTokens := e.Tokens[0:maxLen]
	oTokens := e.Tokens[maxLen:len(e.Tokens)]
	newOffsets := e.Offsets[0:maxLen]
	oOffsets := e.Offsets[maxLen:len(e.Offsets)]
	newSpeToks := e.SpecialTokenMask[0:maxLen]
	oSpeToks := e.SpecialTokenMask[maxLen:len(e.SpecialTokenMask)]
	newAttent := e.AttentionMask[0:maxLen]
	oAttent := e.AttentionMask[maxLen:len(e.AttentionMask)]
	newWords := e.Words[0:maxLen]
	oWords := e.Words[maxLen:len(e.Words)]

	e.Ids = newIds
	e.TypeIds = newTypeIds
	e.Tokens = newTokens
	e.Offsets = newOffsets
	e.SpecialTokenMask = newSpeToks
	e.AttentionMask = newAttent
	e.Words = newWords

	// Separate the overflowing part into as many Encoding as needed
	partSize := maxLen - stride
	overflowing := make([]Encoding, 0)
	partID := 0
	prevEncoding := e

	// while loop
	for int(partSize)*partID < len(oIds) {
		o := Encoding{
			Ids:              getCurrentPart(prevEncoding.Ids, oIds, partSize, partID, stride),
			TypeIds:          getCurrentPart(prevEncoding.TypeIds, oTypeIds, partSize, partID, stride),
			Tokens:           getCurrentPart(prevEncoding.Tokens, oTokens, partSize, partID, stride),
			Offsets:          getCurrentPart(prevEncoding.Offsets, oOffsets, partSize, partID, stride),
			SpecialTokenMask: getCurrentPart(prevEncoding.SpecialTokenMask, oSpeToks, partSize, partID, stride),
			AttentionMask:    getCurrentPart(prevEncoding.AttentionMask, oAttent, partSize, partID, stride),
			Words:            getCurrentPart(prevEncoding.Words, oWords, partSize, partID, stride),
			Overflowing:      make([]Encoding, 0),
		}

		partID++
		overflowing = append(overflowing, o)
		prevEncoding = &overflowing[len(overflowing)-1]
	}

	e.Overflowing = overflowing

	return e, nil
}

// Merge merges all Encodings together
func (e *Encoding) Merge(encodings []Encoding, growingOffsets bool) (retVal *Encoding) {
	retVal = e
	for _, encoding := range encodings {
		retVal = retVal.MergeWith(&encoding, growingOffsets)
	}

	return retVal
}

// MergeWith merges the current encoding with other (pair) encoding
func (e *Encoding) MergeWith(pair *Encoding, growingOffsets bool) (retVal *Encoding) {
	// Merge overflowing
	var overflowings []Encoding
	enOverflowings := e.Overflowing
	penOverflowings := pair.Overflowing

	// 1. All our overflowings with all other overflowings
	for _, o := range enOverflowings {
		nEncoding := o.Clone()
		// 1.1. The pair itself
		merge := nEncoding.MergeWith(pair.Clone(), growingOffsets)
		overflowings = append(overflowings, *merge)

		// 1.2. Its overflowings
		for _, otherO := range penOverflowings {
			nEncoding := o.Clone()
			merge := nEncoding.MergeWith(otherO.Clone(), growingOffsets)
			overflowings = append(overflowings, *merge)
		}
	}

	// 2. Ourself with all the other overflowings
	for _, otherO := range penOverflowings {
		nEncoding := e.Clone()
		merge := nEncoding.MergeWith(otherO.Clone(), growingOffsets)
		overflowings = append(overflowings, *merge)
	}

	e.Overflowing = overflowings

	// Merging others
	originalLen := e.Len()
	if len(pair.SequenceRanges) > 0 {
		for seqID, r := range pair.SequenceRanges {
			start := originalLen + r[0]
			end := originalLen + r[r.Len()-1] + 1
			newRange := NewRange(start, end)
			var oldRange Range
			_ = util.DeepCopy(e.SequenceRanges[seqID], oldRange)
			e.SequenceRanges[seqID] = util.Merge(oldRange, newRange)
		}
	}

	e.Ids = util.Merge(e.Ids, pair.Ids)
	e.Tokens = util.Merge(e.Tokens, pair.Tokens)
	e.Words = util.Merge(e.Words, pair.Words)
	e.TypeIds = util.Merge(e.TypeIds, pair.TypeIds)
	e.SpecialTokenMask = util.Merge(e.SpecialTokenMask, pair.SpecialTokenMask)
	e.AttentionMask = util.Merge(e.AttentionMask, pair.AttentionMask)

	// Offsets
	startingOffset := 0
	offsets := e.Offsets
	if growingOffsets {
		if len(offsets) > 0 {
			last := offsets[len(offsets)-1]
			startingOffset = last[1]
		}
	}

	for _, o := range pair.Offsets {
		adjustedO := []int{
			o[0] + startingOffset,
			o[1] + startingOffset,
		}
		offsets = append(offsets, adjustedO)
	}
	e.Offsets = offsets

	return e
}

// Pad pads current encoding with given length and values to either Left or Right direction.
func (e *Encoding) Pad(targetLength, padID, padTypeID int, padToken string, direction PaddingDirection) *Encoding {
	// 1. Overflowing
	var overflowing []Encoding
	for _, o := range e.Overflowing {
		padded := o.pad(targetLength, padID, padTypeID, padToken, direction)
		overflowing = append(overflowing, *padded)
	}
	e.Overflowing = overflowing

	// 2. Check whether we should pad encoding itself
	// if wanted padding length is smaller, then do nothing
	if len(e.Ids) >= targetLength {
		return e
	}

	paddedEn := e.pad(targetLength, padID, padTypeID, padToken, direction)
	return paddedEn
}

func (e *Encoding) pad(targetLength, padID, padTypeID int, padToken string, direction PaddingDirection) *Encoding {
	padLength := targetLength - len(e.Ids)

	switch direction {
	case Left:
		newIds := make([]int, padLength)
		for i := 0; i < len(newIds); i++ {
			newIds[i] = padID
		}
		newIds = append(newIds, e.Ids...)
		e.Ids = newIds

		newTypeIds := make([]int, padLength)
		for i := 0; i < len(newTypeIds); i++ {
			newTypeIds[i] = padTypeID
		}
		newTypeIds = append(newTypeIds, e.Ids...)
		e.TypeIds = newTypeIds

		newTokens := make([]string, padLength)
		for i := 0; i < len(newTokens); i++ {
			newTokens[i] = padToken
		}
		newTokens = append(newTokens, e.Tokens...)
		e.Tokens = newTokens

		newSpecialTokenMask := make([]int, padLength)
		for i := 0; i < len(newSpecialTokenMask); i++ {
			newSpecialTokenMask[i] = 1
		}
		newSpecialTokenMask = append(newSpecialTokenMask, e.SpecialTokenMask...)
		e.SpecialTokenMask = newSpecialTokenMask

		newAttentionMask := make([]int, padLength)
		for i := 0; i < len(newAttentionMask); i++ {
			newAttentionMask[i] = 0
		}
		newAttentionMask = append(newAttentionMask, e.AttentionMask...)
		e.AttentionMask = newAttentionMask

		newOffsets := make([][]int, padLength)
		for i := 0; i < len(newIds); i++ {
			newOffsets[i] = []int{0, 0}
		}
		newOffsets = append(newOffsets, e.Offsets...)
		e.Offsets = newOffsets

		newWords := make([]int, padLength)
		for i := 0; i < len(newWords); i++ {
			newWords[i] = -1
		}
		newWords = append(newWords, e.Words...)
		e.Words = newWords

	case Right:
		for range padLength {
			e.Ids = append(e.Ids, padID)
			e.TypeIds = append(e.TypeIds, padTypeID)
			e.Tokens = append(e.Tokens, padToken)
			e.SpecialTokenMask = append(e.SpecialTokenMask, 1)
			e.AttentionMask = append(e.AttentionMask, 0)
			e.Offsets = append(e.Offsets, []int{0, 0})
			e.Words = append(e.Words, -1)
		}
	}

	return e
}

// Token2Sequence returns the index of the sequence containing the given token.
func (e *Encoding) Token2Sequence(token int) (int, bool) {
	if token > e.Len() {
		return -1, false
	} else if len(e.SequenceRanges) == 0 {
		return 0, true
	} else {
		for seqID, r := range e.SequenceRanges {
			if r.Contains(token) {
				return seqID, true
			}
		}

		return -1, false
	}
}

// SequenceRange returns the range to target to retrieve something (word id, offsets, ...)
// related to the given sequence id.
func (e *Encoding) SequenceRange(sequenceID int) (Range, error) {
	r, ok := e.SequenceRanges[sequenceID]
	if !ok {
		err := fmt.Errorf("input sequence_id is out of range")
		return nil, err
	}

	return r[0:e.Len()], nil
}

func getCurrentPart[T any](previous, current []T, size, idx, stride int) []T {
	if size <= 0 || idx < 0 || stride < 0 {
		return nil
	}

	if len(previous) < stride {
		stride = len(previous)
	}

	start := idx * size
	if start >= len(current) {
		return previous[len(previous)-stride:]
	}

	end := min((idx+1)*size, len(current))

	curr := current[start:end]
	prev := previous[len(previous)-stride:]

	return append(prev, curr...)
}
