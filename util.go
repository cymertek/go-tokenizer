package tokenizer

import (
	"errors"
	"log"
	"slices"
	"sync"
)

// TruncationParams configures how encodings are truncated to a maximum length.
type TruncationParams struct {
	// MaxLength is the target maximum number of tokens after truncation.
	MaxLength int
	// Strategy determines which encoding(s) to truncate (OnlyFirst, OnlySecond, LongestFirst).
	Strategy TruncationStrategy
	// Stride is the overlap in tokens between consecutive truncated parts during overflow handling.
	Stride int
}

// PaddingParams configures how encodings are padded to a target length.
type PaddingParams struct {
	// Strategy determines padding strategy (BatchLongest or Fixed).
	Strategy PaddingStrategy
	// Direction specifies whether to pad Left or Right.
	Direction PaddingDirection
	// PadID is the token ID used for padding.
	PadID int
	// PadTypeID is the type ID assigned to each padding token.
	PadTypeID int
	// PadToken is the string representation of the pad token.
	PadToken string
}

// PaddingStrategy describes how padding length is determined: either dynamically based on the longest encoding in a batch, or at a fixed size.
type PaddingStrategy struct {
	Value any
	Name  string
}

// PaddingStrategyOption configures a PaddingStrategy (e.g., WithBatchLongest, WithFixed).
type PaddingStrategyOption func(*PaddingStrategy)

// WithBatchLongest configures padding to use the longest encoding in a batch as the target length.
func WithBatchLongest() PaddingStrategyOption {
	return func(ps *PaddingStrategy) {
		ps.Value = "BatchLongest"
		ps.Name = "BatchLongest"
	}
}

// WithFixed configures padding to use a specific fixed length for all encodings in a batch.
func WithFixed(size int) PaddingStrategyOption {
	return func(ps *PaddingStrategy) {
		ps.Value = size
		ps.Name = "Fixed"
	}
}

// NewPaddingStrategy creates a new PaddingStrategy with default BatchLongest behavior, optionally overridden by options.
func NewPaddingStrategy(opts ...PaddingStrategyOption) *PaddingStrategy {
	const defaultVal = "BatchLongest"

	ps := &PaddingStrategy{
		Value: defaultVal,
		Name:  defaultVal,
	}

	for _, opt := range opts {
		opt(ps)
	}

	return ps

}

// TruncationStrategy is enum of int type represents truncation strategy
type TruncationStrategy int

const (
	// LongestFirst truncates the longest encoding first when sharing budget between sequences.
	LongestFirst TruncationStrategy = iota
	// OnlyFirst truncates only the first sequence, leaving the second unchanged.
	OnlyFirst
	// OnlySecond truncates only the second sequence, leaving the first unchanged.
	OnlySecond
)

const (
	// SecondSequenceNotProvided is returned when truncation targets a pair but no second encoding was given.
	SecondSequenceNotProvided = "truncation error: second sequence not provided"
	// SequenceTooShort is returned when a sequence is shorter than the requested truncation length.
	SequenceTooShort = "truncation error: sequence to truncate too short to respect the provided max_length"
)

// TruncateEncodings reduces encoding lengths to fit within params.MaxLength, distributing removal according to Strategy.
func TruncateEncodings(encoding, pairEncoding *Encoding, params *TruncationParams) (tEncoding, tPairEncoding *Encoding) {
	var (
		totalLength int
		toRemove    int
		err         error
	)

	if params.MaxLength == 0 {
		return encoding, pairEncoding
	}

	totalLength = len(encoding.GetIds())
	if pairEncoding != nil {
		totalLength = len(encoding.GetIds()) + len(pairEncoding.GetIds())
	}

	if totalLength < params.MaxLength {
		return encoding, pairEncoding
	}

	toRemove = totalLength - params.MaxLength

	switch params.Strategy {
	case LongestFirst:
		nFirst := len(encoding.GetIds())
		nSecond := 0
		if pairEncoding != nil {
			nSecond = len(pairEncoding.GetIds())
		}

		for i := 0; i < toRemove; i++ {
			if nFirst > nSecond {
				nFirst--
			}
			nSecond--
		}

		_, _ = encoding.Truncate(nFirst, params.Stride)
		if pairEncoding != nil {
			_, _ = pairEncoding.Truncate(nSecond, params.Stride)
		}

	case OnlyFirst, OnlySecond:
		var truncateFunc = func(target *Encoding) (*Encoding, error) {
			targetLength := len(target.GetIds())
			if targetLength > toRemove {
				_, _ = target.Truncate(targetLength-toRemove, params.Stride)
				return target, nil
			} else {
				err := errors.New(SequenceTooShort)
				return nil, err
			}
		}

		if params.Strategy == OnlyFirst {
			encoding, err = truncateFunc(encoding)
		} else if pairEncoding != nil {
			pairEncoding, err = truncateFunc(pairEncoding)
		} else {
			err = errors.New(SecondSequenceNotProvided)
		}

	}

	if err != nil {
		log.Fatal(err)
	}

	return encoding, pairEncoding
}

// PadEncodings pads a batch of encodings to the same length using PaddingParams.
func PadEncodings(encodings []Encoding, params PaddingParams) []Encoding {
	if len(encodings) == 0 {
		return encodings
	}

	var padLength int

	switch params.Strategy.Name {
	case "Fixed":
		padLength = params.Strategy.Value.(int)
	case "BatchLongest":
		max := 0
		for _, encoding := range encodings {
			if len(encoding.GetIds()) > max {
				max = len(encoding.GetIds())
			}
		}
		padLength = max
	}

	// Pad encodings concurrently using goroutines for better performance on large batches.
	var wg sync.WaitGroup
	wg.Add(len(encodings))
	results := make([]Encoding, len(encodings))
	for i, e := range encodings {
		go func(idx int, enc Encoding) {
			defer wg.Done()
			paddedEn := enc.Pad(padLength, params.PadID, params.PadTypeID, params.PadToken, params.Direction)
			if paddedEn != nil {
				results[idx] = *paddedEn
			}
		}(i, e)
	}
	wg.Wait()

	return results
}

// Range is a contiguous sequence of token positions in an encoding.
type Range []int

// NewRange creates a new range from start to end (exclusive).
func NewRange(start, end int) Range {
	if start < 0 {
		panic("Invalid 'start' for NewRange()")
	}
	if end < 0 || end <= start {
		panic("Invalid 'end' for NewRange()")
	}

	var r []int
	for i := start; i < end; i++ {
		r = append(r, i)
	}

	return r
}

// Len returns the number of elements in this range.
func (r Range) Len() int {
	return len(r)
}

// Contains checks if an item is present in the range.
func (r Range) Contains(item int) bool {
	return slices.Contains(r, item)
}

// IsEmpty returns true if this range contains no elements.
func (r Range) IsEmpty() bool {
	return len(r) == 0
}

// Min returns the minimum value in the range, or 0 if empty.
func (r Range) Min() int {
	if len(r) == 0 {
		return 0
	}
	m := r[0]
	for _, v := range r[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

// Max returns the maximum value in the range, or 0 if empty.
func (r Range) Max() int {
	if len(r) == 0 {
		return 0
	}
	m := r[0]
	for _, v := range r[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// First returns the first element, or 0 if empty.
func (r Range) First() int {
	if len(r) == 0 {
		return 0
	}
	return r[0]
}

// Last returns the last element, or 0 if empty.
func (r Range) Last() int {
	if len(r) == 0 {
		return 0
	}
	return r[len(r)-1]
}

// Clone returns a new independent copy of the range.
func (r Range) Clone() Range {
	c := make(Range, len(r))
	copy(c, r)
	return c
}
