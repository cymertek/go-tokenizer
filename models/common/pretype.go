package common

import "strings"

// PreType identifies the pre-tokenization strategy for BPE tokenizers.
type PreType int

const (
	PreDefault PreType = iota
	PreGPT2
	PreQwen2
	PreLlama3
	PreStarcoder
	PreDeepSeekLLM
	PreFalcon
	PreQwen35
	PreStableLM2
	PreGPT4O
	PreGemma4
)

// PreTypeFromString maps a GGUF tokenizer.ggml.pre string to a PreType.
func PreTypeFromString(s string) PreType {
	switch s {
	case "qwen2", "megrez":
		return PreQwen2
	case "gpt-2", "phi-2", "jina-es", "jina-de", "jina-v2-es", "jina-v2-de",
		"a.x-4.0":
		return PreGPT2
	case "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1",
		"pixtral", "midm-2.0", "lfm2", "jina-v5-nano":
		return PreLlama3
	case "starcoder", "refact", "command-r", "smollm", "codeshell",
		"exaone", "minerva-7b":
		return PreStarcoder
	case "deepseek-llm":
		return PreDeepSeekLLM
	case "deepseek-coder":
		return PreDeepSeekLLM
	case "deepseek-v3", "hunyuan-dense", "joyai-llm":
		return PreDeepSeekLLM
	case "falcon":
		return PreFalcon
	case "qwen35":
		return PreQwen35
	case "stablelm2", "hunyuan", "solar-open":
		return PreStableLM2
	case "gpt-4o", "llama4", "kanana2":
		return PreGPT4O
	case "gemma4":
		return PreGemma4
	case "mpt", "olmo", "jais", "trillion", "granite-docling":
		return PreGPT2
	default:
		return PreDefault
	}
}

// stripMERank removes the optional rank suffix from merge strings.
func stripMERank(s string) string {
	if i := strings.Index(s, " # rank:"); i >= 0 {
		return s[:i]
	}
	return s
}
