package models

// This file is intentionally minimal. All shared types (TokenizerData, PreType,
// Tokenizer interface), the registry map, and registration functions live in
// models/common/. The root package re-exports them via type aliases in types.go
// so existing callers of gguf.Register() and gguf.New() continue to work.
