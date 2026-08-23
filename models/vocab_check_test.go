package models

import (
	"fmt"
	"testing"
	"unicode/utf8"
)

func TestVocabSpacePrefix(t *testing.T) {
	path := "/workdir/Bonsai-8B.gguf"
	data, err := ReadTokenizerFromGGUF(path)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}

	fmt.Printf("Model: %s\n", data.Model)
	fmt.Printf("PreType: %d\n", data.PreType)
	fmt.Printf("Tokens count: %d\n", len(data.Tokens))

	// Find space-prefixed tokens (Ġ = U+0120)
	count := 0
	for _, tok := range data.Tokens {
		if len(tok) > 0 {
			r, _ := utf8.DecodeRuneInString(tok)
			if r == 'Ġ' { // Ġ = U+0120
				count++
				if count <= 5 {
					fmt.Printf("Space-prefixed token: %q\n", tok)
				}
			}
		}
	}
	fmt.Printf("\nTotal space-prefixed tokens (Ġ): %d\n", count)

	// Check for specific tokens we saw in debugging
	for _, tok := range data.Tokens {
		if tok == "main" || tok == "func" || tok == "Print" || tok == "ln" ||
			tok == "Ġmain" || tok == "Ġfunc" || tok == "ĠPrint" || tok == "Ġln" {
			fmt.Printf("Found exact match: %q\n", tok)
		}
	}

	// Check if there are any tokens starting with spaceChar (whatever was auto-detected)
	if data.PreType > 0 {
		fmt.Printf("\nPreType value: %d\n", data.PreType)
	}
}
