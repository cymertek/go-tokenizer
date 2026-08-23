package tokenizer

import (
	"fmt"
	"log"
	"os"
)

var (
	// CachedDir is the directory where tokenizer cache data is stored.
	CachedDir string = "NOT_SETTING"
	// tokenizerEnvKey is the environment variable used to override the cache directory path.
	tokenizerEnvKey string = "GO_TOKENIZER"
)

func initCachePath() {
	// default path: {$HOME}/.cache/tokenizer
	homeDir := os.Getenv("HOME")
	CachedDir = fmt.Sprintf("%s/.cache/tokenizer", homeDir)

	initEnv()

	log.Printf("INFO: CachedDir=%q\n", CachedDir)
}

func initEnv() {
	val := os.Getenv(tokenizerEnvKey)
	if val != "" {
		CachedDir = val
	}

	if _, err := os.Stat(CachedDir); os.IsNotExist(err) {
		if err := os.MkdirAll(CachedDir, 0755); err != nil {
			log.Fatal(err)
		}
	}
}
