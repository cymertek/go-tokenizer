package bpe

import (
	"runtime"
	"sync"
	"unicode/utf8"
)

const ChunkSize = 16 * 1024 // 16KB — ConcurrentThreshold defined in bpe.go

type chunkResult struct {
	ids []int
	idx int
}

// encodeConcurrent splits large inputs into chunks and encodes them in parallel.
func (b *BPE) encodeConcurrent(text string) []int {
	chunks := b.splitIntoChunks(text)
	if len(chunks) == 1 {
		return b.encodeIDsInner(chunks[0].text)
	}

	sem := make(chan struct{}, runtime.GOMAXPROCS(0)*4)
	results := make([]chunkResult, len(chunks))
	var wg sync.WaitGroup

	for i, ch := range chunks {
		wg.Add(1)
		go func(idx int, c chunk) {
			sem <- struct{}{}
			defer func() {
				<-sem
				wg.Done()
			}()
			results[idx] = chunkResult{
				ids: b.encodeFragmentText(c.text),
				idx: idx,
			}
		}(i, ch)
	}

	wg.Wait()
	return b.stitchResultsConcurrent(results)
}

// encodeFragmentText encodes text as if it were a single pre-tokenized fragment.
func (b *BPE) encodeFragmentText(text string) []int {
	if id, ok := b.vocab[text]; ok {
		return []int{id}
	}
	// Pre-tokenize the chunk and encode each fragment
	splits := preTokenize(text, b.preType, b.spaceChar)
	var ids []int
	for _, s := range splits {
		fragIDs := b.encodeFragment(s.Text)
		if len(fragIDs) == 0 && s.Text != "" {
			fragIDs = b.encodeCharacters(s.Text)
		}
		ids = append(ids, fragIDs...)
	}
	return ids
}

type chunk struct {
	text string
}

func (b *BPE) splitIntoChunks(text string) []chunk {
	if len(text) <= ConcurrentThreshold {
		return []chunk{{text: text}}
	}

	var chunks []chunk
	i := 0
	for i < len(text) {
		end := i + ChunkSize
		if end >= len(text) {
			chunks = append(chunks, chunk{text: text[i:]})
			break
		}

		// Ensure split point is at a rune boundary (not in the middle of multi-byte UTF-8).
		_, w := utf8.DecodeRuneInString(text[end:])
		end += w

		chunks = append(chunks, chunk{text: text[i:end]})
		i = end
	}
	return chunks
}

func (b *BPE) stitchResultsConcurrent(results []chunkResult) []int {
	totalLen := 0
	for _, r := range results {
		totalLen += len(r.ids)
	}
	final := make([]int, 0, totalLen)
	for _, r := range results {
		final = append(final, r.ids...)
	}
	return final
}
