package common

// Constructor is a function that creates a Tokenizer instance from the given
// TokenizerData. Each tokenizer sub-package (bpe, spm, unigram, wordpiece) provides
// its own constructor via init() and registers it with Register().
type Constructor func(data *TokenizerData) (Tokenizer, error)

// registry maps model type strings ("bpe", "spm", "unigram", "wordpiece") to their
// constructor functions. Populated by init() in each sub-package during package loading.
var registry = make(map[string]Constructor)

// Register adds a tokenizer constructor for the given model type. Called automatically
// by init() in each sub-package (bpe, spm, unigram, wordpiece). Panics if a constructor
// is already registered for the same model type — use this only during package initialization.
//
// Example (in sub-package init()):
//
//	func init() {
//	    Register("spm", func(data *TokenizerData) (Tokenizer, error) {
//	        return New(data)
//	    })
//	}
func Register(modelType string, fn Constructor) {
	if _, exists := registry[modelType]; exists {
		panic("gguf: tokenizer already registered for model " + modelType)
	}
	registry[modelType] = fn
}

// RegisteredTypes returns a slice of all currently registered tokenizer model type strings.
// The order is not guaranteed. Use this to validate data.Model before calling New().
func RegisteredTypes() []string {
	types := make([]string, 0, len(registry))
	for k := range registry {
		types = append(types, k)
	}
	return types
}
