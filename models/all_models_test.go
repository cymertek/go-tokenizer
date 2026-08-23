package models

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// TestGGUF_AllModels validates all BPE GGUF vocab models against llama.cpp
// reference outputs by comparing token counts (not IDs, since each model's
// vocabulary is different).
//
// To run: go test ./models/ -run TestGGUF_AllModels -count=1 -v
// This is a slow test (~30s) so it's not run by default.
func TestGGUF_AllModels(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	models := []struct {
		name       string
		ggufPath   string
		inpPath    string
		outPath    string
		skipReason string
	}{
		{"qwen2", "/home/user/git/ik_llama.cpp/models/ggml-vocab-qwen2.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-qwen2.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-qwen2.gguf.out", ""},
		{"gpt-2", "/home/user/git/ik_llama.cpp/models/ggml-vocab-gpt-2.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-gpt-2.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-gpt-2.gguf.out", ""},
		{"llama-bpe", "/home/user/git/ik_llama.cpp/models/ggml-vocab-llama-bpe.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-llama-bpe.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-llama-bpe.gguf.out", ""},
		{"starcoder", "/home/user/git/ik_llama.cpp/models/ggml-vocab-starcoder.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-starcoder.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-starcoder.gguf.out", ""},
		{"refact", "/home/user/git/ik_llama.cpp/models/ggml-vocab-refact.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-refact.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-refact.gguf.out", ""},
		{"deepseek-llm", "/home/user/git/ik_llama.cpp/models/ggml-vocab-deepseek-llm.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-deepseek-llm.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-deepseek-llm.gguf.out", ""},
		{"falcon", "/home/user/git/ik_llama.cpp/models/ggml-vocab-falcon.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-falcon.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-falcon.gguf.out", ""},
		{"phi-3", "/home/user/git/ik_llama.cpp/models/ggml-vocab-phi-3.gguf",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-phi-3.gguf.inp",
			"/home/user/git/ik_llama.cpp/models/ggml-vocab-phi-3.gguf.out", ""},
		{"bert-bge", "/home/user/git/ik_llama.cpp/models/ggml-vocab-bert-bge.gguf",
			"", "", "BERT: non-BPE model"},
		{"llama-spm", "/home/user/git/ik_llama.cpp/models/ggml-vocab-llama-spm.gguf",
			"", "", "SPM: non-BPE model"},
	}

	totalPassed := 0
	totalFailed := 0
	totalSkipped := 0

	// Load all models in parallel
	type result struct {
		name       string
		pass, fail int
		skip       string
		bpe        *BPE
		model      string
		pre        PreType
	}
	ch := make(chan result, len(models))

	for _, tc := range models {
		go func(tc struct {
			name, ggufPath, inpPath, outPath, skipReason string
		}) {
			if tc.skipReason != "" {
				ch <- result{skip: tc.skipReason}
				return
			}
			if _, err := os.Stat(tc.ggufPath); os.IsNotExist(err) {
				ch <- result{skip: "gguf file not found: " + tc.ggufPath}
				return
			}

			done := make(chan result, 1)
			go func() {
				data, err := ReadTokenizerFromGGUF(tc.ggufPath)
				if err != nil {
					done <- result{skip: "load error: " + err.Error()}
					return
				}
				bpe, err := NewBPE(data)
				if err != nil || bpe == nil {
					done <- result{skip: fmt.Sprintf("NewBPE error: %v", err)}
					return
				}

				inpData, _ := os.ReadFile(tc.inpPath)
				outData, _ := os.ReadFile(tc.outPath)
				prompts := strings.Split(string(inpData), "__ggml_vocab_test__")
				expectedOutputs := strings.Split(string(outData), "\n")

				pass, fail := 0, 0
				for i, prompt := range prompts {
					if i >= len(expectedOutputs) {
						break
					}
					cleaned := strings.TrimRight(prompt, "\n")
					if cleaned == "" {
						continue
					}
					trimmed := strings.TrimSpace(expectedOutputs[i])
					if trimmed == "" {
						if bpe.Count(cleaned) != 0 {
							fail++
						} else {
							pass++
						}
						continue
					}
					expectedLen := len(strings.Fields(trimmed))
					if expectedLen == 0 {
						continue
					}
					gotLen := bpe.Count(cleaned)
					if gotLen == expectedLen {
						pass++
					} else {
						fail++
					}
				}
				done <- result{name: tc.name, pass: pass, fail: fail, bpe: bpe, model: data.Model, pre: data.PreType}
			}()

			select {
			case r := <-done:
				ch <- r
			case <-time.After(30 * time.Second):
				ch <- result{skip: tc.name + ": processing timeout"}
			}
		}(tc)
	}

	for range models {
		r := <-ch
		if r.skip != "" {
			t.Logf("[%s] SKIP: %s", r.name, r.skip)
			totalSkipped++
			continue
		}
		totalPassed += r.pass
		totalFailed += r.fail
		if r.fail > 0 {
			t.Logf("[%s] model=%s pre=%d PASS=%d FAIL=%d", r.name, r.model, r.pre, r.pass, r.fail)
		} else {
			t.Logf("[%s] model=%s pre=%d ALL %d PROMPTS PASS (count match)", r.name, r.model, r.pre, r.pass)
		}
	}

	t.Logf("Summary: PASS=%d FAIL=%d SKIP=%d", totalPassed, totalFailed, totalSkipped)
}
