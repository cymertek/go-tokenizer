package processor

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/cymertek/go-tokenizer"
	"github.com/cymertek/go-tokenizer/util"
)

type SequenceEnum int

const (
	A SequenceEnum = iota
	B
)

type Piece interface {
	// ExtractId(s string) Piece
	WithTypeId(typeID int)
}

type SequencePiece struct {
	Id     SequenceEnum `json:"id"`
	TypeId int          `json:"type_id"`
}

var _ Piece = new(SequencePiece)

type SpecialTokenPiece struct {
	Id     string `json:"id"`
	TypeId int    `json:"type_id"`
}

var _ Piece = new(SpecialTokenPiece)

func extractId(s string) (Piece, error) {
	var p Piece
	if rest, ok := strings.CutPrefix(s, "$"); ok {

		var isNum bool
		num, err := strconv.Atoi(rest)
		if err == nil {
			isNum = true
		}

		switch {
		case rest == "", rest == "A", rest == "a":
			p = &SequencePiece{
				Id:     A,
				TypeId: 0,
			}
		case rest == "B", rest == "b":
			p = &SequencePiece{
				Id:     B,
				TypeId: 0,
			}

		case isNum:
			// Numbers are treated as TypeIds on sequence A.
			p = &SequencePiece{Id: A, TypeId: num}

		default:
			err := fmt.Errorf("cannot extract id from input %q", s)
			return nil, err
		}
	} else {
		p = &SpecialTokenPiece{
			Id:     s,
			TypeId: 0,
		}
	}

	return p, nil
}

func NewSequencePiece(id string, typeID int) *SequencePiece {
	var seqEnum SequenceEnum
	if id == "A" {
		seqEnum = A
	} else {
		seqEnum = B
	}
	return &SequencePiece{
		Id:     seqEnum,
		TypeId: typeID,
	}
}

func NewSpecialTokenPiece(id string, typeID int) *SpecialTokenPiece {
	return &SpecialTokenPiece{
		Id:     id,
		TypeId: typeID,
	}
}

// Implement Piece for SequencePiece:
// ----------------------------------
func (p *SequencePiece) WithTypeId(v int) {
	p.TypeId = v
}

func (p *SpecialTokenPiece) WithTypeId(v int) {
	p.TypeId = v
}

func NewPiece(s string) (Piece, error) {
	parts := strings.Split(s, ":")

	var (
		p   Piece
		err error
	)
	switch len(parts) {
	case 2:
		typeID, err := strconv.Atoi(parts[1])
		if err != nil {
			err = fmt.Errorf("cannot build piece from string %q", s)
			return nil, err
		}

		p, err = extractId(parts[0])
		if err != nil {
			err = fmt.Errorf("%w. Cannot build Piece from string %q", err, s)
			return nil, err
		}

		p.WithTypeId(typeID)

	case 1:
		p, err = extractId(parts[0])
		if err != nil {
			err = fmt.Errorf("%w. Cannot build Piece from string %q", err, s)
			return nil, err
		}

	default:
		err = fmt.Errorf("cannot build piece from string %q", s)
		return nil, err
	}

	return p, nil
}

// Represents a bunch of tokens to be used in a template.
// Usually, special tokens have only one associated id/token but in
// some cases, it might be interesting to have multiple ids/tokens.
type SpecialToken struct {
	// A unique id used to identify this SpecialToken in the template
	Id string

	// The list of associated ids
	Ids []int

	// The list of associated tokens
	Tokens []string
}

func NewSpecialToken(id string, ids []int, tokens []string) *SpecialToken {
	return &SpecialToken{
		Id:     id,
		Ids:    ids,
		Tokens: tokens,
	}
}

func NewSpecialTokenFrom(s string, id int) *SpecialToken {
	return NewSpecialToken(s, []int{id}, []string{s})
}

type Template []Piece

func NewTemplateFromOne(s string) (Template, error) {
	parts := strings.Split(s, " ")

	return NewTemplateFromMulti(parts)
}

func NewTemplateFromMulti(parts []string) (Template, error) {
	var tpl []Piece
	for _, part := range parts {
		p, err := NewPiece(part)
		if err != nil {
			return nil, err
		}
		tpl = append(tpl, p)
	}

	return tpl, nil
}

func NewTemplate(v any) (Template, error) {
	switch typ := v.(type) {
	case string:
		return NewTemplateFromOne(v.(string))
	case []string:
		return NewTemplateFromMulti(v.([]string))
	default:
		err := fmt.Errorf("unsupported input type %v", typ)
		return nil, err
	}
}

// A bunch of [`SpecialToken`] represented by their ID.
type Tokens struct {
	TokenMap    map[string]SpecialToken // NOTE. HF is an ordered map
	orderedKeys []string                // order of the TokenMap
}

func DefaultTokens() *Tokens {
	return &Tokens{
		TokenMap:    make(map[string]SpecialToken),
		orderedKeys: nil,
	}
}

func NewTokensFrom(toks []SpecialToken) *Tokens {
	m := make(map[string]SpecialToken)
	var keys []string
	for _, tok := range toks {
		keys = append(keys, tok.Id)
		m[tok.Id] = tok
	}

	return &Tokens{
		TokenMap:    m,
		orderedKeys: keys,
	}
}

func NewTokensFromMap(m map[string]SpecialToken) *Tokens {
	// Sort keys for deterministic ordering — Go map iteration is non-deterministic,
	// but callers rely on GetItemByOrder returning stable results.
	var keys []string
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	return &Tokens{
		TokenMap:    m,
		orderedKeys: keys,
	}
}

func NewTokens(toks []tokenizer.Token) *Tokens {
	m := make(map[string]SpecialToken)
	var keys []string
	for _, tok := range toks {
		spt := NewSpecialTokenFrom(tok.Value, tok.Id)
		keys = append(keys, tok.Value)
		m[tok.Value] = *spt
	}

	return &Tokens{
		TokenMap:    m,
		orderedKeys: keys,
	}
}

func (t *Tokens) GetItemByOrder(index int) (SpecialToken, bool) {
	k := t.orderedKeys[index]

	return t.GetItemByKey(k)
}

func (t *Tokens) GetItemByKey(id string) (SpecialToken, bool) {
	val, ok := t.TokenMap[id]
	return val, ok
}

// / This PostProcessor takes care of processing each input `Encoding` by applying
// / the corresponding template, before merging them in the final Encoding.
// /
// / A `Template` is actually a sequence of `Piece` that will be
// / concatenated together in the given order. Each `Piece` represents either
// / one of the input `Encoding` or a `SpecialToken`.
// /
// / ## Example
// / ```
// / # use tokenizers::processors::template::TemplateProcessing;
// / let template = TemplateProcessing::builder()
// /     .try_single("[CLS] $A [SEP]").unwrap()
// /     .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1").unwrap()
// /     .special_tokens(vec![("[CLS]", 1), ("[SEP]", 0)])
// /     .build()
// /     .unwrap();
// / ```
// /
type TemplateProcessing struct {
	Single        Template
	Pair          Template
	AddedSingle   int
	AddedPair     int
	SpecialTokens *Tokens
}

type TemplateProcessingDeserializer struct {
	Single        Template
	Pair          Template
	SpecialTokens *Tokens
}

func NewTemplateProcessingFrom(t *TemplateProcessingDeserializer) *TemplateProcessing {
	addedSingle := countAdded(t.Single, t.SpecialTokens)
	addedPair := countAdded(t.Pair, t.SpecialTokens)

	return &TemplateProcessing{
		Single:        t.Single,
		Pair:          t.Pair,
		AddedSingle:   addedSingle,
		AddedPair:     addedPair,
		SpecialTokens: t.SpecialTokens,
	}
}

func NewTemplateProcessing(single, pair Template, specialTokens *Tokens) *TemplateProcessing {
	return NewTemplateProcessingFrom(&TemplateProcessingDeserializer{
		Single:        single,
		Pair:          pair,
		SpecialTokens: specialTokens,
	})
}

// Count the number of added tokens in the given template
func countAdded(container Template, specialTokens *Tokens) int {
	var count int
	for _, p := range container {
		typ := util.GetType(p)
		switch typ {
		case "*SequencePiece":
			count += 0
		case "*SpecialTokenPiece":
			spt := p.(*SpecialTokenPiece)
			id := spt.Id
			specialToken, ok := specialTokens.GetItemByKey(id)
			if ok {
				count += len(specialToken.Ids)
			}
		default:
			msg := fmt.Sprintf("Unsupported typ %q for 'specialTokens' item\n", typ)
			panic(msg)
		}
	}

	return count
}

type TemplateProcessingBuilder struct {
	*TemplateProcessing
}

func (b *TemplateProcessingBuilder) updateAddedTokens() {
	b.AddedSingle = countAdded(b.Single, b.SpecialTokens)
	b.AddedPair = countAdded(b.Pair, b.SpecialTokens)
}

func (b *TemplateProcessingBuilder) NewSingle(v any) {
	tpl, err := NewTemplate(v)
	if err != nil {
		panic("err")
	}

	b.Single = tpl
	b.updateAddedTokens()
}

func (b *TemplateProcessingBuilder) NewPair(v any) {
	tpl, err := NewTemplate(v)
	if err != nil {
		panic("err")
	}

	b.Pair = tpl
	b.updateAddedTokens()
}

func (b *TemplateProcessingBuilder) NewSpecialTokens(tokens []tokenizer.Token) {
	b.SpecialTokens = NewTokens(tokens)
	b.updateAddedTokens()
}

func (b *TemplateProcessingBuilder) DefaultAdded(isSingle bool) int {
	var t Template
	if isSingle {
		t = b.Single
	} else {
		t = b.Pair
	}

	return countAdded(t, b.SpecialTokens)
}

func (b *TemplateProcessingBuilder) Validate() error {
	var (
		hasA bool
		hasB bool
	)

	for _, piece := range b.Pair {
		if piece.(*SequencePiece).Id == A {
			hasA = true
		}

		if piece.(*SequencePiece).Id == B {
			hasB = true
		}
	}

	pairHasBoth := hasA && hasB

	if !pairHasBoth {
		err := fmt.Errorf("pair template must use both sequences")
		return err
	}

	check := func(sp string) string {
		var exist bool
		tok, ok := b.SpecialTokens.GetItemByOrder(0)
		if !ok {
			exist = false
		} else {
			exist = util.Contains(tok.Tokens, sp)
		}

		if exist {
			return sp
		} else {
			return ""
		}
	}

	var missing []string
	var pieces []Piece = append(b.Single, b.Pair...)
	for _, p := range pieces {
		typ := util.GetType(p)
		switch typ {
		case "*SequencePiece":
			// None
		case "*SpecialTokenPiece":
			id := p.(*SpecialTokenPiece).Id
			s := check(id)
			if s != "" {
				missing = append(missing, s)
			}
		}
	}

	if len(missing) > 0 {
		var msg string
		for _, s := range missing {
			v := fmt.Sprintf("Missing SpecialToken %q", s)
			msg = fmt.Sprintf("%s, %s", msg, v)
		}

		return fmt.Errorf("%s", msg)
	}

	return nil
}

func DefaultTemplateProcessing() *TemplateProcessing {
	single, err := NewTemplateFromOne("$0")
	if err != nil {
		panic(err)
	}

	pair, err := NewTemplateFromOne("$1")
	if err != nil {
		panic(err)
	}

	specialTokens := DefaultTokens()

	return &TemplateProcessing{
		Single:        single,
		Pair:          pair,
		AddedSingle:   0,
		AddedPair:     0,
		SpecialTokens: specialTokens,
	}
}

func (tp *TemplateProcessing) Builder() *TemplateProcessingBuilder {
	return &TemplateProcessingBuilder{tp}
}

func (tp *TemplateProcessingBuilder) Build() *TemplateProcessing {
	return tp.TemplateProcessing
}

func (tp *TemplateProcessing) ApplyTemplate(template []Piece, encodings []tokenizer.Encoding, addSpecialTokens bool) []tokenizer.Encoding {
	var finalEncodings []tokenizer.Encoding

	for _, piece := range template {
		typ := util.GetType(piece)

		switch typ {
		case "*SequencePiece":
			sp := piece.(*SequencePiece)
			id := sp.Id
			typeID := sp.TypeId
			i := 0
			if id != A {
				i = 1
			}
			encoding := encodings[id]
			typeIds := util.Repeat(typeID, encoding.Len())
			encoding.SetTypeIds(typeIds)
			encoding.SetSequenceIds(i)

			finalEncodings = append(finalEncodings, encoding)

			if len(encoding.Overflowing) > 0 && len(template) == 3 {
				var processed []tokenizer.Encoding
				for _, ov := range encoding.Overflowing {
					ov.SetTypeIds(util.Repeat(typeID, ov.Len()))
					ov.SetSequenceIds(i)
					result := tp.ApplyTemplate(template, []tokenizer.Encoding{ov}, addSpecialTokens)
					if len(result) > 0 {
						processed = append(processed, result[0])
					}
				}
				val := finalEncodings[len(finalEncodings)-1]
				val.Overflowing = processed
				finalEncodings[len(finalEncodings)-1] = val
			}

		case "*SpecialTokenPiece":
			spt := piece.(*SpecialTokenPiece)
			id := spt.Id
			typeID := spt.TypeId
			if addSpecialTokens {
				tok, ok := tp.SpecialTokens.GetItemByKey(id)
				if !ok {
					msg := fmt.Sprintf("Token not found with key %q", id)
					panic(msg)
				}
				length := len(tok.Ids)

				ids := tok.Ids
				typeIds := util.Repeat(typeID, length)
				tokens := tok.Tokens
				offsets := util.Repeat([]int{0, 0}, length)
				specialTokenMask := util.Repeat(1, length)
				attentionMask := util.Repeat(1, length)
				var overflowing []tokenizer.Encoding = nil
				encoding := tokenizer.NewEncoding(ids, typeIds, tokens, offsets, specialTokenMask, attentionMask, overflowing)

				finalEncodings = append(finalEncodings, *encoding)
			}
		}
	}

	return finalEncodings
}

// Implement PostProcessor for TemplateProcessing:
// -----------------------------------------------

var _ tokenizer.PostProcessor = new(TemplateProcessing)

func (tp *TemplateProcessing) AddedTokens(isPair bool) int {
	if isPair {
		return tp.AddedPair
	}

	return tp.AddedSingle
}

func (tp *TemplateProcessing) Process(encoding, pairEncoding *tokenizer.Encoding, addSpecialTokens bool) *tokenizer.Encoding {
	encodings := tokenizer.PrepareEncodings(encoding, pairEncoding)
	var template Template
	switch len(encodings) {
	case 2:
		template = tp.Pair
	case 1:
		template = tp.Single
	default:
		panic("Shouldn't be here. 'encoding' must be != nil")
	}

	if len(template) == 3 {
		return tp.processSingle(encoding, template, addSpecialTokens)
	}

	// Pair mode: build base encoding by merging main × pair_base, then construct
	// overflow entries manually to match BERT's structure:
	//   - entry_0: main_overflow merged with pair_base (with nested overflow from each pair_chunk)
	//   - entries 1..N: combinations of main/base chunks with pair chunks
	mainResult := tp.applyPairSide(encoding, template, A, true, true)
	pairBase := tp.applyPairSide(pairEncoding, template, B, true, true)

	result := mainResult.Clone()
	result.MergeWith(&pairBase, false)

	// Collect all overflow combinations using the original encodings' Overflowing fields.
	if len(mainResult.Overflowing) > 0 || len(pairBase.Overflowing) == 0 {
		var overflowings []tokenizer.Encoding

		// Step 1: For each main overflow chunk, merge with full pair result (creates entry_0).
		for _, mChunk := range encoding.Overflowing {
			mainEnc := tp.applyPairSideForMainEnc(&mChunk, template, A)
			entry := mainEnc.Clone()
			entry.MergeWith(&pairBase, false)

			// Step 1.2: For each pair overflow chunk, create a nested entry (main_overflow × pair_chunk).
			for _, pChunk := range pairEncoding.Overflowing {
				pairChunkEnc := tp.applyPairSide(&pChunk, template, B, false, false)
				nestedEntry := mainEnc.Clone()
				nestedResult := nestedEntry.MergeWith(&pairChunkEnc, false)
				entry.Overflowing = append(entry.Overflowing, *nestedResult)
			}

			overflowings = append(overflowings, *entry)
		}

		// Step 2: For each pair overflow chunk, create standalone entries.
		cleanBase := *mainResult.Clone()
		for _, pChunk := range pairEncoding.Overflowing {
			pairChunkEnc := tp.applyPairSide(&pChunk, template, B, false, false)

			// Entry: main_overflow × pair_chunk (standalone).
			for _, mChunk := range encoding.Overflowing {
				mainEnc := tp.applyPairSideForMainEnc(&mChunk, template, A)
				entry := mainEnc.Clone()
				entry.MergeWith(&pairChunkEnc, false)
				overflowings = append(overflowings, *entry)
			}

			// Entry: main_base × pair_chunk (with nested overflow if there are main overflows).
			baseEntry := cleanBase
			baseEntry.MergeWith(&pairChunkEnc, false)

			// Add nested overflow entries (main_overflow × this pair_chunk) to base entry.
			for _, mChunk := range encoding.Overflowing {
				mainEnc := tp.applyPairSideForMainEnc(&mChunk, template, A)
				nestedEntry := mainEnc.Clone()
				nestedResult := nestedEntry.MergeWith(&pairChunkEnc, false)
				baseEntry.Overflowing = append(baseEntry.Overflowing, *nestedResult)
			}

			overflowings = append(overflowings, baseEntry)
		}

		result.Overflowing = overflowings
	}

	return result
}

func (tp *TemplateProcessing) processSingle(encoding *tokenizer.Encoding, template Template, _ bool) *tokenizer.Encoding {
	result := tp.buildPairSideEncoding(encoding, template, A, 0, true, true, true)

	var overflowings []tokenizer.Encoding
	for _, ov := range encoding.Overflowing {
		chunk := tp.buildPairSideEncoding(&ov, template, A, 0, true, false, false)
		overflowings = append(overflowings, chunk)
	}
	result.Overflowing = overflowings

	return &result
}

func (tp *TemplateProcessing) applyPairSide(encoding *tokenizer.Encoding, template Template, seqPieceId SequenceEnum, includeSpecialTokens bool, isBaseEncoding bool) tokenizer.Encoding {
	typeID := 0
	for _, p := range template {
		if sp, ok := p.(*SequencePiece); ok && sp.Id == seqPieceId {
			typeID = sp.TypeId
			break
		}
	}

	result := tp.buildPairSideEncoding(encoding, template, seqPieceId, typeID, includeSpecialTokens, isBaseEncoding, false)
	return result
}

// applyPairSideForMainEnc builds the main encoding for an overflow chunk (mainEnc).
// It includes structural pieces that match the side's TypeId only — CLS and SEP_A for side A (typeID=0),
// SEP_B for side B (typeID=1). This prevents duplicate structural tokens when merging with pairChunk/pairBase.
func (tp *TemplateProcessing) applyPairSideForMainEnc(encoding *tokenizer.Encoding, template Template, seqPieceId SequenceEnum) tokenizer.Encoding {
	typeID := 0
	for _, p := range template {
		if sp, ok := p.(*SequencePiece); ok && sp.Id == seqPieceId {
			typeID = sp.TypeId
			break
		}
	}

	result := tp.buildPairSideEncoding(encoding, template, seqPieceId, typeID, false, false, false)
	return result
}

// buildPairSideEncoding builds an encoding by assembling pieces from the pair template that belong to one sequence side.
func (tp *TemplateProcessing) buildPairSideEncoding(encoding *tokenizer.Encoding, template Template, seqPieceId SequenceEnum, typeID int, includeSpecialTokens bool, isBaseEncoding bool, includeAllStructural bool) tokenizer.Encoding {

	var ids []int
	var typeIds []int
	var tokens []string
	var offsets [][]int
	var specialTokenMask []int
	var attentionMask []int

	for _, p := range template {
		switch piece := p.(type) {
		case *SpecialTokenPiece:
			if tp.shouldIncludeStructural(piece, seqPieceId, typeID, includeSpecialTokens, isBaseEncoding, includeAllStructural) {
				tok, ok := tp.SpecialTokens.GetItemByKey(piece.Id)
				if !ok {
					panic(fmt.Sprintf("Token not found with key %q", piece.Id))
				}
				for i, id := range tok.Ids {
					ids = append(ids, id)
					typeIds = append(typeIds, typeID)
					tokens = append(tokens, tok.Tokens[i])
					offsets = append(offsets, []int{0, 0})
					specialTokenMask = append(specialTokenMask, 1)
					attentionMask = append(attentionMask, 1)
				}
			}

		case *SequencePiece:
			if piece.Id == seqPieceId {
				for j := 0; j < encoding.Len(); j++ {
					ids = append(ids, encoding.GetIds()[j])
					typeIds = append(typeIds, typeID)
					tokens = append(tokens, encoding.GetTokens()[j])
					offsets = append(offsets, encoding.GetOffsets()[j])
					specialTokenMask = append(specialTokenMask, 0)
					attentionMask = append(attentionMask, 1)
				}
			}
		}
	}

	result := tokenizer.NewEncoding(ids, typeIds, tokens, offsets, specialTokenMask, attentionMask, nil)

	// Set SequenceRanges from TypeIds (excluding special token positions).
	if len(result.TypeIds) > 0 && len(result.SequenceRanges) == 0 {
		result.BuildRangesFromTypeIds(0, false)
	}

	return *result
}

// shouldIncludeStructural determines whether a SpecialTokenPiece should be included.
// Rules:
//   - mainEnc/pairBase/pairChunk (includeAllStructural=false): only includes pieces whose actual TypeId matches the
//     sequence's typeID to prevent duplication during merge with the other side.
func (tp *TemplateProcessing) shouldIncludeStructural(piece *SpecialTokenPiece, _ SequenceEnum, typeID int, _ bool, _ bool, includeAllStructural bool) bool {
	// mainEnc on each side includes all structural pieces from the template.
	if includeAllStructural {
		return true
	}

	// pairBase/pairChunk only include pieces matching the sequence type's typeID.
	// CLS(TypeId=0) goes to side A (typeID=0), second SEP(TypeId=1) goes to side B (typeID=1).
	return piece.TypeId == typeID
}


