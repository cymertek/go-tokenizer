package bpe

// byteTrie is a byte-level trie optimized for GGUF BPE vocabulary matching.
// It walks the trie byte-by-byte (no string slicing) and returns the longest
// terminal token ID match, giving O(k) per token where k is the matched length.
type byteTrie struct {
	root trieNode
}

type trieNode struct {
	endID    int // -1 if not end of a token
	children [256]*trieNode
}

// newByteTrie builds a trie from token strings with their IDs.
func newByteTrie(tokens []string) *byteTrie {
	tr := &byteTrie{}
	for idx, tok := range tokens {
		insert(&tr.root, []byte(tok), idx)
	}
	return tr
}

func insert(node *trieNode, b []byte, id int) {
	if len(b) == 0 {
		return
	}
	if node.children[b[0]] == nil {
		node.children[b[0]] = &trieNode{endID: -1} // Initialize endID to -1 (no token)
	}
	if len(b) == 1 {
		node.children[b[0]].endID = id
		return
	}
	insert(node.children[b[0]], b[1:], id)
}

// matchLongest walks the trie byte-by-byte, returning the longest terminal
// token ID and its byte-length. Returns (-1, 0) if no match.
func (tr *byteTrie) matchLongest(b []byte) (id, length int) {
	id, length = -1, 0
	node := &tr.root
	for i := 0; i < len(b); i++ {
		c := b[i]
		if node.children[c] == nil {
			break
		}
		node = node.children[c]
		if node.endID >= 0 {
			id = node.endID
			length = i + 1 // Track actual byte position, not terminal count
		}
	}
	return id, length
}

