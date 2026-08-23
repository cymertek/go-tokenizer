package unigram

// Byte-level trie for O(k) longest-match vocabulary lookup.
// Replaces linear scan of tokens with a single pass through the text.

type trieNode struct {
	endID    int // -1 if not terminal (end of a token), else the piece ID
	children [256]*trieNode
}

type byteTrie struct {
	root trieNode
	size int
}

func newByteTrie() *byteTrie {
	return &byteTrie{}
}

// insert adds a token string with its ID into the trie.
func (tr *byteTrie) insert(s string, id int) {
	cur := &tr.root
	for i := 0; i < len(s); i++ {
		b := byte(s[i])
		if cur.children[b] == nil {
			tr.size++
			cur.children[b] = &trieNode{endID: -1}
		}
		cur = cur.children[b]
	}
	cur.endID = id
}

// matchLongest finds the longest prefix of b that is a token in the trie.
// Returns (id, length) — id=-1 if no match found.
func (tr *byteTrie) matchLongest(b []byte) (int, int) {
	cur := &tr.root
	bestID := -1
	bestLen := 0
	for i := 0; i < len(b); i++ {
		bb := b[i]
		if cur.children[bb] == nil {
			break
		}
		cur = cur.children[bb]
		if cur.endID >= 0 {
			bestID = cur.endID
			bestLen = i + 1
		}
	}
	return bestID, bestLen
}

// has checks whether the exact string s is a token in the trie.
func (tr *byteTrie) has(s string) bool {
	cur := &tr.root
	for i := 0; i < len(s); i++ {
		bb := byte(s[i])
		if cur.children[bb] == nil {
			return false
		}
		cur = cur.children[bb]
	}
	return cur.endID >= 0
}
