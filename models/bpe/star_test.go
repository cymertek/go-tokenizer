package bpe

import (
	"fmt"
	"testing"
)

func TestStarCoderPreTokenize(t *testing.T) {
	text := "it's a test"

	// Debug: step by step through composeRules
	fmt.Printf("Input: %q\n", text)

	// Step 1: modeGPT2 on the full text
	step1 := modeGPT2(text)
	fmt.Printf("modeGPT2: %v\n", fragmentTexts(step1))

	// Full preTokenize
	full := preTokenize(text, PreStarcoder, 0)
	fmt.Printf("preTokenize: %v\n", fragmentTexts(full))
}

func fragmentTexts(frags []PreSplit) []string {
	var out []string
	for _, f := range frags {
		out = append(out, f.Text)
	}
	return out
}
