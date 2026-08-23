package tokenizer

import (
	"log"
	"sort"
	"unicode"
	"unicode/utf8"

	norm "github.com/cymertek/go-tokenizer/normalizer"
)

// AddedToken represents a token added by the user on top of the
// existing model vocabulary.
//
// AddedToken can be configured to specify the behaviour they should
// have in various situations. I.e.,:
// - Whether they should only match single words
// - Whether to include any whitespace on its left or right
type AddedToken struct {
	// Content is the content of added token
	Content string
	// whether this token is single word or break words
	SingleWord bool
	// Whether this token should strip whitespace on its left
	LStrip bool
	// Whether this token should strip whitespace on its right
	RStrip bool
	// Whether this token should be normalized
	Normalized bool
}

// DefaultAddedToken initiates a default AddedToken
func DefaultAddedToken() (retVal AddedToken) {
	return AddedToken{
		Content:    "",
		SingleWord: false,
		LStrip:     false,
		RStrip:     false,
		Normalized: true,
	}
}

// ATOption configures an AddedToken's behavior (single word, strip whitespace, normalization).
type ATOption func(at *AddedToken)

// WithSingleWord specifies whether this token should only match on whole single words.
func WithSingleWord(singleWord bool) ATOption {
	return func(at *AddedToken) {
		at.SingleWord = singleWord
	}
}

// WithLStrip specifies whether this token should strip leading whitespace when matching.
func WithLStrip(lstrip bool) ATOption {
	return func(at *AddedToken) {
		at.LStrip = lstrip
	}
}

// WithRStrip specifies whether this token should strip trailing whitespace when matching.
func WithRStrip(rstrip bool) ATOption {
	return func(at *AddedToken) {
		at.RStrip = rstrip
	}
}

// WithNormalized specifies whether this token should be normalized before matching against input text.
func WithNormalized(normalized bool) ATOption {
	return func(at *AddedToken) {
		at.Normalized = normalized
	}
}

// NewAddedToken builds an AddedToken from given content
// specifying whether it is intended to be a special token.
// NOTE. Special token ar not normalized by default.
func NewAddedToken(s string, special bool, opts ...ATOption) (retVal AddedToken) {
	addedTok := DefaultAddedToken()
	addedTok.Content = s
	addedTok.Normalized = !special

	for _, opt := range opts {
		opt(&addedTok)
	}

	return addedTok
}

// SetSingleWord configures whether this token should only match on whole single words.
func (at AddedToken) SetSingleWord(singleWord bool) (retVal AddedToken) {
	at.SingleWord = singleWord
	return at
}

// SetLStrip configures whether this token should strip leading whitespace when matching.
func (at AddedToken) SetLStrip(lstrip bool) (retVal AddedToken) {
	at.LStrip = lstrip
	return at
}

// SetRStrip configures whether this token should strip trailing whitespace when matching.
func (at AddedToken) SetRStrip(rstrip bool) (retVal AddedToken) {
	at.RStrip = rstrip
	return at
}

// SetNormalized configures whether this token should be normalized before matching.
func (at AddedToken) SetNormalized(normalized bool) (retVal AddedToken) {
	at.Normalized = normalized
	return at
}

// trieNode is a node in the added-token trie.
// The trie is built rune-by-rune to correctly handle multi-byte UTF-8.
type trieNode struct {
	end   []int              // terminal token indices
	child [256]*trieNode     // ASCII child (rune < 0x80)
	uni   map[rune]*trieNode // non-ASCII child
}

// atTrie holds the trie and the token list.
type atTrie struct {
	root   trieNode
	tokens []AddedToken
}

// build inserts all tokens into the trie, rune-by-rune.
func (t *atTrie) build(tokens []AddedToken) {
	t.tokens = tokens
	for i, tok := range tokens {
		n := &t.root
		for _, r := range tok.Content {
			if r < 0x80 {
				if n.child[r] == nil {
					n.child[r] = &trieNode{uni: make(map[rune]*trieNode)}
				}
				n = n.child[r]
			} else {
				if n.uni == nil {
					n.uni = make(map[rune]*trieNode)
				}
				if n.uni[r] == nil {
					n.uni[r] = &trieNode{uni: make(map[rune]*trieNode)}
				}
				n = n.uni[r]
			}
		}
		n.end = append(n.end, i)
	}
}

// findMatches finds all added tokens in sentence using the trie.
// Returns id-offset pairs covering the entire string, with id=-1 for unmatched regions.
func (t *atTrie) findMatches(sentence string) []idOffsets {
	if len(sentence) == 0 {
		return []idOffsets{{id: -1, offsets: []int{0, 0}}}
	}

	sRunes := []rune(sentence)
	n := len(sRunes)
	var matches []idOffsets

	for i := 0; i < n; i++ { //nolint:rangeint // needs index for byte offset computation
		node := &t.root

		// Compute the byte offset of rune i (the match start position).
		startByte := 0
		for k := 0; k < i; k++ {
			startByte += utf8.RuneLen(sRunes[k])
		}

		for j := i; j < n; j++ {
			r := sRunes[j]

			var next *trieNode
			if r < 0x80 {
				next = node.child[r]
			} else {
				next = node.uni[r]
			}
			if next == nil {
				break
			}
			node = next
			_ = startByte + utf8.RuneLen(r) // byteEnd was replaced by tokenLen

			if len(node.end) > 0 {
				for _, id := range node.end {
					innerTok := t.tokens[id]
					bounded := true
					if innerTok.SingleWord {
						if i > 0 {
							prevRune := sRunes[i-1]
							if unicode.IsLetter(prevRune) || unicode.IsDigit(prevRune) {
								bounded = false
							}
						}
						if bounded && j+1 < n {
							nextRune := sRunes[j+1]
							if unicode.IsLetter(nextRune) || unicode.IsDigit(nextRune) {
								bounded = false
							}
						}
					}

					if !bounded {
						continue
					}

					// Use full token content length, not just the last rune
					tokenLen := 0
					for _, tr := range innerTok.Content {
						tokenLen += utf8.RuneLen(tr)
					}
					match := idOffsets{id: id, offsets: []int{startByte, startByte + tokenLen}}

					// LStrip: include preceding whitespace.
					if innerTok.LStrip {
						s := match.offsets[0]
						for s > 0 {
							rn, sz := utf8.DecodeLastRuneInString(sentence[:s])
							if !unicode.IsSpace(rn) {
								break
							}
							s -= sz
						}
						match.offsets[0] = s
					}

					// RStrip: include following whitespace.
					if innerTok.RStrip {
						e := match.offsets[1]
						for e < len(sentence) {
							rn, sz := utf8.DecodeRuneInString(sentence[e:])
							if !unicode.IsSpace(rn) {
								break
							}
							e += sz
						}
						match.offsets[1] = e
					}

					matches = append(matches, match)
				}
			}
		}
	}

	return resolveOverlaps(matches, n)
}

// resolveOverlaps handles overlapping matches by keeping the one with the lowest index.
func resolveOverlaps(matches []idOffsets, n int) []idOffsets {

	if len(matches) == 0 {
		return []idOffsets{{id: -1, offsets: []int{0, 0}}}
	}

	sort.Slice(matches, func(i, j int) bool {
		if matches[i].offsets[0] != matches[j].offsets[0] {
			return matches[i].offsets[0] < matches[j].offsets[0]
		}
		// When offsets are the same, prefer the longer match (higher id typically means longer token).
		iLen := matches[i].offsets[1] - matches[i].offsets[0]
		jLen := matches[j].offsets[1] - matches[j].offsets[0]
		if iLen != jLen {
			return iLen > jLen
		}
		return matches[i].id < matches[j].id
	})

	var splits []idOffsets
	currentOffset := 0

	for _, m := range matches {
		if m.offsets[0] < currentOffset {
			continue
		}

		// Gap before this match.
		if currentOffset < m.offsets[0] {
			splits = append(splits, idOffsets{id: -1, offsets: []int{currentOffset, m.offsets[0]}})
		}

		splits = append(splits, m)
		currentOffset = m.offsets[1]
	}

	// Gap after last match.
	if currentOffset < n {
		splits = append(splits, idOffsets{id: -1, offsets: []int{currentOffset, n}})
	}

	return splits
}

// AddedVocabulary is a vocabulary built on top of the Model
//
// This provides a way to add new vocabulary to a Tokenizer that has already been trained,
// in a previous process, maybe by someone else. This is especially interesting in the case
// of fine-tunings, where we want to finetune a model while adding some new functionalities
// using some new special tokens, or maybe add some tokens in the case of unknown tokens, etc.
//
// One of the reasons we need to handle these tokens outside of the model is simply that
// for many models, it is not possible to add new tokens after the training process. For example,
// using BPE, the training process generates merges pairs along the vocabulary, and any token
// in the vocabulary can be decomposed in other tokens, down to the original alphabet. If we
// were to add new tokens after this training process, we couldn't make sure the merges pairs
// exist as required.
type AddedVocabulary struct {
	// Contains the mapping from String (token content) to ID. This map contains both special
	// tokens and classic added tokens that were added to the this vocabulary.
	addedTokenMap map[string]int
	// Contains the mapping from ID to AddedToken for all the added tokens, both special
	// and classic.
	addedTokenMapR map[int]string
	// Contains only the classic AddedToken, in the specific order the user gave them.
	addedTokens []AddedToken
	// Contains only the special AddedToken, in the specific order the user gave them.
	specialTokens []AddedToken
	// A map, containing all the special token for easy access while decoding. This let's
	// us remove them easily with an O(1) complexity.
	specialTokensSet map[string]bool
	// Tries for fast matching of added tokens (replaces regexp/regexpset).
	splitTrie           atTrie
	splitNormalizedTrie atTrie
}

// NewAddedVocabulary creates a new empty AddedVocabulary ready for tokens to be added.
func NewAddedVocabulary() (retVal AddedVocabulary) {
	return AddedVocabulary{
		addedTokenMap:       make(map[string]int, 0),
		addedTokenMapR:      make(map[int]string, 0),
		addedTokens:         []AddedToken{},
		specialTokens:       []AddedToken{},
		specialTokensSet:    make(map[string]bool, 0),
		splitTrie:           atTrie{},
		splitNormalizedTrie: atTrie{},
	}
}

// Len returns the number of tokens in this added vocabulary.
func (av *AddedVocabulary) Len() int {
	return len(av.addedTokenMap)
}

// GetVocab returns the mapping from token content strings to their assigned IDs.
func (av *AddedVocabulary) GetVocab() (retVal map[string]int) {
	return av.addedTokenMap
}

// TokenToID returns the ID of a token string in this vocabulary, or delegates to the model if not found.
func (av *AddedVocabulary) TokenToID(token string, model Model) (retVal int, ok bool) {
	retVal, ok = av.addedTokenMap[token]
	if !ok {
		return model.TokenToID(token)
	}
	return retVal, ok
}

// IDToToken returns the token content string for a given ID in this vocabulary.
func (av *AddedVocabulary) IDToToken(id int, model Model) (retVal string, ok bool) {
	retVal, ok = av.addedTokenMapR[id]
	if !ok {
		return model.IDToToken(id)
	}
	return retVal, ok
}

// IsSpecialToken returns true if the given token content is registered as a special token.
func (av *AddedVocabulary) IsSpecialToken(token string) bool {
	_, ok := av.specialTokensSet[token]
	return ok
}

// AddSpecialTokens adds the given tokens to the special token set and then delegates to AddTokens.
// Returns the number of tokens actually added.
func (av *AddedVocabulary) AddSpecialTokens(tokens []AddedToken, model Model, normalizer norm.Normalizer) (retVal int) {
	for _, tok := range tokens {
		_, isExist := av.specialTokensSet[tok.Content]
		if tok.Content != "" && !isExist {
			av.specialTokens = append(av.specialTokens, tok)
			av.specialTokensSet[tok.Content] = true
		}
	}
	return av.AddTokens(tokens, model, normalizer)
}

// AddTokens adds the given tokens to this vocabulary and rebuilds internal tries.
// Returns the number of tokens actually added (excluding duplicates).
func (av *AddedVocabulary) AddTokens(tokens []AddedToken, model Model, normalizer norm.Normalizer) (retVal int) {
	ignored := 0
	for _, token := range tokens {
		if token.Content == "" {
			ignored++
			continue
		}

		var id int
		if i, ok := av.TokenToID(token.Content, model); ok {
			ignored++
			id = i
		} else {
			id = model.GetVocabSize() + len(av.addedTokenMap)
			av.addedTokenMap[token.Content] = id

			if _, ok := av.specialTokensSet[token.Content]; !ok {
				av.addedTokens = append(av.addedTokens, token)
			}
		}

		av.addedTokenMapR[id] = token.Content
	}

	av.refreshAddedTokens(normalizer)

	return len(tokens) - ignored
}

// refreshAddedTokens reconstructs our internal tries when new tokens are added to the vocabulary.
//
// We build two tries from the same token list:
//   - splitTrie: stores non-normalized tokens (original Content)
//   - splitNormalizedTrie: stores normalized tokens (content after normalizer)
//
// Both tries use the same indices, so trie matches map to the same token in both tries.
func (av *AddedVocabulary) refreshAddedTokens(nm norm.Normalizer) {
	// Build tokens list ordered by map ID (so trie indices match map IDs).
	// This is critical: trie node.end stores slice indices, which must match
	// the IDs in addedTokenMap/addedTokenMapR.
	tokens := make([]AddedToken, av.Len())

	// Build a map from content to full AddedToken metadata (preserves SingleWord, LStrip, etc.)
	contentToTok := make(map[string]AddedToken)
	for _, t := range av.specialTokens {
		contentToTok[t.Content] = t
	}
	for _, t := range av.addedTokens {
		contentToTok[t.Content] = t
	}
	for i := 0; i < av.Len(); i++ {
		content := av.addedTokenMapR[i]
		if tok, ok := contentToTok[content]; ok {
			tokens[i] = tok
		} else {
			tokens[i] = AddedToken{Content: content}
		}
	}

	// Build non-normalized trie (original content).
	av.splitTrie.build(tokens)

	// Build normalized trie (content after normalizer).
	var normTokens []AddedToken
	for _, tok := range tokens {
		tokCopy := tok
		if nm != nil {
			ns := norm.NewNormalizedFrom(tokCopy.Content)
			normalized, _ := nm.Normalize(ns)
			if normalized != nil {
				tokCopy.Content = normalized.GetNormalized()
			}
		}
		normTokens = append(normTokens, tokCopy)
	}
	av.splitNormalizedTrie.build(normTokens)
}

type idOffsets struct {
	id      int // optional - None value = -1
	offsets []int
}

// findMatches finds any AddedToken in the given sentence, using the provided trie.
func (av *AddedVocabulary) findMatches(sentence string, trie atTrie) []idOffsets {
	return trie.findMatches(sentence)
}

type SplitIdx struct {
	Normalized *norm.NormalizedString
	Tokens     []Token
}

// splitWithIndices splits the input sentence to extract anything found from the trie, as well as
// the list of corresponding IDs.
func (av *AddedVocabulary) splitWithIndices(sentence *norm.NormalizedString, trie atTrie) []SplitIdx {

	ioPairs := av.findMatches(sentence.GetNormalized(), trie)

	var splits []SplitIdx

	for _, p := range ioPairs {
		slice := sentence.Slice(norm.NewRange(p.offsets[0], p.offsets[1], norm.NormalizedTarget))
		if p.id == -1 {
			splits = append(splits, SplitIdx{slice, nil})
		} else {
			value := slice.GetNormalized()
			length := len(value)
			split := SplitIdx{slice, []Token{NewToken(p.id, value, []int{0, length})}}
			splits = append(splits, split)
		}
	}

	return splits
}

// ExtractAndNormalize extracts added tokens from the given sentence, normalizing both the input and token content along the way.
// Some tokens should match against their normalized representation as well as the non-normalized one. For example, when expecting
// to extract the token `yesterday` in the input sentence `I read a book Yesterday`, if the normalizer lowercases everything, we expect a match.
func (av *AddedVocabulary) ExtractAndNormalize(sequence string, n norm.Normalizer) *PreTokenizedString {

	pretokenized := NewPreTokenizedString(sequence)

	// 1. Extract all non-normalized tokens from the non-normalized string
	pretok1 := pretokenized.Split(func(idx int, seq *norm.NormalizedString) []SplitIdx {
		return av.splitWithIndices(seq, av.splitTrie)
	})

	// 2. Extract normalized tokens from normalized pieces of the string
	pretok2 := pretok1.Split(func(i int, seq *norm.NormalizedString) []SplitIdx {
		newSeq := seq
		var err error
		if n != nil {
			newSeq, err = n.Normalize(seq)
			if err != nil {
				log.Fatal(err)
			}
		}
		result := av.splitWithIndices(newSeq, av.splitNormalizedTrie)
		// Preserve content for gap pieces (no normalized tokens found).
		for j, s := range result {
			if s.Normalized != nil && s.Normalized.GetNormalized() == "" && seq.GetNormalized() != "" {
				result[j].Normalized = seq
			}
		}
		return result
	})

	return pretok2
}

// AddedTokenWithID wraps an added token with its assigned ID and whether it is a special token.
type AddedTokenWithID struct {
	Id      int        // Id assigned to this token (note: follows HuggingFace convention).
	Special bool       // Whether this is a special token.
	Token   AddedToken // The target AddedToken.
}

// (Serialize was removed — use Tokenizer.Serialize(io.Writer) instead.)
