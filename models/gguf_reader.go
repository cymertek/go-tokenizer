// Package models provides GGUF file reading utilities for extracting tokenizer data.
package models

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	"github.com/cymertek/go-tokenizer/models/common"
)

// ReadTokenizerFromGGUF reads tokenizer metadata from a GGUF file.
func ReadTokenizerFromGGUF(path string) (*common.TokenizerData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf file %q: %w", path, err)
	}
	defer f.Close()

	data := &common.TokenizerData{}

	// Read and verify GGUF header (24 bytes)
	var header [24]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, fmt.Errorf("read gguf header: %w", err)
	}

	if string(header[0:4]) != "GGUF" {
		return nil, fmt.Errorf("invalid GGUF magic: got %q, want %q", header[0:4], "GGUF")
	}

	version := binary.LittleEndian.Uint32(header[4:8])
	nKV := binary.LittleEndian.Uint64(header[8:16])

	if version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	fileSize, _ := f.Seek(0, io.SeekEnd)
	f.Seek(24, io.SeekStart) // back to start of KV section

	// Read KV section with position tracking
	// Note: using uint64 for large GGUF files (may have >2^31 KV entries)
	for i := uint64(0); i < nKV; i++ { //nolint:rangeint // uint64 needed for large file support
		posBefore, _ := f.Seek(0, io.SeekCurrent)
		keyName, valueBytes, err := readKVEntry(f)
		if err != nil {
			if strings.Contains(err.Error(), "unexpected EOF") || err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read kv[%d]: %w", i, err)
		}
		posAfterRead, _ := f.Seek(0, io.SeekCurrent)

		// Skip any tensor metadata that follows weight entries
		// In GGUF v3, tensors have metadata (n_dims, shape, etc.) after the value
		var posAfter int64
		posAfter = skipTensorMetadata(f, keyName, posAfterRead, fileSize)

		// seek to where skipTensorMetadata found us (if it advanced past current position)
		if posAfter > posAfterRead {
			posAfter, err = f.Seek(posAfter, io.SeekStart)
			if err != nil {
				return nil, fmt.Errorf("seek to %d: %w", posAfter, err)
			}
		}

		consumed := posAfter - posBefore

		if consumed == 0 {
			return nil, fmt.Errorf("kv[%d]: readKVEntry returned 0 bytes consumed for key %q — possible parse error", i, keyName)
		}

		processKVValue(data, keyName, valueBytes)
	}


	return data, nil
}

// skipTensorMetadata skips any tensor metadata bytes that follow weight entries in GGUF v3.
// It scans forward from the current position until it finds a valid key_len pattern (1-256 bytes)
// with a clean ASCII name and valid btype (0-12).
func skipTensorMetadata(f *os.File, currentKey string, startPos int64, fileSize int64) int64 {
	// Only skip metadata for tensor weight entries (keys ending with ".weight" or ".bias")
	if !strings.HasSuffix(currentKey, ".weight") && !strings.HasSuffix(currentKey, ".bias") {
		return startPos
	}

	pos := startPos + 1 // Start scanning AFTER the current position (not including it)


	// Scan forward up to 2000 bytes looking for a valid KV entry
	for pos < fileSize-12 && pos <= startPos+2000 {
		// Read key_len (8 bytes) at current position
		keyLenBytes := make([]byte, 8)
		n, err := f.ReadAt(keyLenBytes, pos)
		if err != nil || n < 8 {
			break
		}

		keyLen := binary.LittleEndian.Uint64(keyLenBytes)

		// Debug: print positions near the expected next entry (5926889)
		if pos >= 5926880 && pos <= 5926900 {
		}

		// Valid key_len should be 1-256 bytes for a KV entry name
		if keyLen >= 1 && keyLen <= 256 {
			nameStart := pos + 8
			nameEnd := nameStart + int64(keyLen)
			btypePos := nameEnd

			// Check if we have room for btype after the name
			if btypePos+4 <= fileSize {
				// Read the name bytes
				nameBytes := make([]byte, keyLen)
				n2, err2 := f.ReadAt(nameBytes, nameStart)
				if err2 != nil || n2 < int(keyLen) {
					pos++
					continue
				}

				name := string(nameBytes)

				// Read btype
				btypeBytes := make([]byte, 4)
				n3, err3 := f.ReadAt(btypeBytes, btypePos)
				if err3 != nil || n3 < 4 {
					pos++
					continue
				}

				btype := binary.LittleEndian.Uint32(btypeBytes)

				if btype > 12 {
					pos++
					continue
				}

				// Validate the name: should be clean ASCII with no embedded binary data
				isCleanName := true
				for _, b := range nameBytes {
					if b < 32 || b > 126 {
						isCleanName = false
						break
					}
				}

				if !isCleanName {
					pos++
					continue
				}

				// Valid KV entry names should start with lowercase letter, underscore, or common prefix
				if len(name) > 0 && (name[0] >= 'a' && name[0] <= 'z') {
					// Found what looks like a valid KV entry — stop skipping
					return pos
				}

			} else {
			}
		}

		pos++
	}

	// Didn't find valid KV entry — return original position (parser will handle error)
	if pos > startPos {
	} else {
	}
	return startPos
}

// readKVEntry reads a single KV entry from the GGUF file and returns (key, raw_value_bytes).
func readKVEntry(f *os.File) (string, []byte, error) {
	// Read key length (8 bytes)
	keyLenBytes := make([]byte, 8)
	n, err := f.Read(keyLenBytes)
	if err != nil || n < 8 {
		return "", nil, fmt.Errorf("read key_len: got %d/%d bytes, err=%v", n, 8, err)
	}
	keyLen := binary.LittleEndian.Uint64(keyLenBytes)

	// Sanity check for reasonable key length (catches alignment garbage)
	if keyLen > 1024*1024 { // max 1MB key name
		return "", nil, fmt.Errorf("key_len too large: %d", keyLen)
	}

	// Read key
	keyData := make([]byte, keyLen)
	n, err = io.ReadFull(f, keyData)
	if err != nil || uint64(n) < keyLen {
		return "", nil, fmt.Errorf("read key: got %d/%d bytes, err=%v", n, keyLen, err)
	}
	keyName := string(keyData)

	// Read value type (4 bytes)
	btypeBytes := make([]byte, 4)
	if _, err := f.Read(btypeBytes); err != nil {
		return "", nil, err
	}
	btype := binary.LittleEndian.Uint32(btypeBytes)

	switch btype {
	case 0: // BTypeUint8 (1 byte)
		valBytes := make([]byte, 1)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 1: // BTypeInt8 (1 byte)
		valBytes := make([]byte, 1)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 2: // BTypeUint16 (2 bytes)
		valBytes := make([]byte, 2)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 3: // BTypeInt16 (2 bytes)
		valBytes := make([]byte, 2)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 4: // BTypeUint32 (4 bytes)
		valBytes := make([]byte, 4)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 5: // BTypeInt32 (4 bytes)
		valBytes := make([]byte, 4)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 6: // BTypeFloat32 (4 bytes)
		valBytes := make([]byte, 4)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 7: // BTypeBool (1 byte)
		valBytes := make([]byte, 1)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 8: // BTypeString
		strLenBytes := make([]byte, 8)
		if _, err := f.Read(strLenBytes); err != nil {
			return "", nil, err
		}
		strLen := binary.LittleEndian.Uint64(strLenBytes)

		valData := make([]byte, strLen)
		if _, err := io.ReadFull(f, valData); err != nil {
			return "", nil, err
		}
		return keyName, valData, nil

	case 9: // BTypeArray
		elemTypeBytes := make([]byte, 4)
		countBytes := make([]byte, 8)
		if _, err := f.Read(elemTypeBytes); err != nil {
			return "", nil, err
		}
		if _, err := f.Read(countBytes); err != nil {
			return "", nil, err
		}

		elemType := binary.LittleEndian.Uint32(elemTypeBytes)
		arrayCount := binary.LittleEndian.Uint64(countBytes)

		var allData []byte

		switch elemType {
		case 8: // String array - preserve length prefixes for downstream parsing
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				slenBytes := make([]byte, 8)
				if _, err := f.Read(slenBytes); err != nil {
					return "", nil, fmt.Errorf("read string[%d] len: %w", j, err)
				}
				slen := binary.LittleEndian.Uint64(slenBytes)

				if slen > 10*1024*1024 { // sanity check: max 10MB per token
					return "", nil, fmt.Errorf("string[%d] len too large: %d", j, slen)
				}

				sdata := make([]byte, slen)
				if _, err := io.ReadFull(f, sdata); err != nil {
					return "", nil, fmt.Errorf("read string[%d]: %w", j, err)
				}

				// Preserve length prefix (uint32 LE) + data for downstream parsing
				lenPrefix := make([]byte, 4)
				binary.LittleEndian.PutUint32(lenPrefix, uint32(slen))
				allData = append(allData, lenPrefix...)
				allData = append(allData, sdata...)

				if j < 5 || j == arrayCount-1 {
				} else if j == 5 {
				}
			}

		case 0: // Uint8 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 1)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		case 4: // Uint32 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 4)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		case 5: // Int32 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 4)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		case 6: // Float32 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 4)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		case 10: // Uint64 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 8)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		case 11: // Int64 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 8)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		case 12: // Float64 array
			for j := uint64(0); j < arrayCount; j++ { //nolint:rangeint
				valBytes := make([]byte, 8)
				if _, err := f.Read(valBytes); err != nil {
					return "", nil, err
				}
				allData = append(allData, valBytes...)
			}

		default:
			return "", nil, fmt.Errorf("unsupported array element type %d", elemType)
		}

		return keyName, allData, nil

	case 10: // BTypeUint64 (8 bytes)
		valBytes := make([]byte, 8)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 11: // BTypeInt64 (8 bytes)
		valBytes := make([]byte, 8)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	case 12: // BTypeFloat64 (8 bytes)
		valBytes := make([]byte, 8)
		if _, err := f.Read(valBytes); err != nil {
			return "", nil, err
		}
		return keyName, valBytes, nil

	default:
		// Skip unsupported types gracefully - they're not tokenizer-related
		return keyName, []byte{}, nil
	}
}

// processKVValue processes extracted KV values and populates TokenizerData fields.
func processKVValue(data *common.TokenizerData, keyName string, rawBytes []byte) {
	switch keyName {
	case "general.architecture":
		if len(rawBytes) > 0 && rawBytes[0] >= 'a' && rawBytes[0] <= 'z' {
			data.Model = mapGGUFArchitecture(string(rawBytes))
		}

	case "tokenizer.ggml.tokens":
		tokens, count := parseStringArray(rawBytes)
		if count > 1000000 {
			fmt.Printf("ERROR: token count too large: %d (possible parse error)\n", count)
			return
		}
		data.Tokens = tokens

	case "tokenizer.ggml.merges":
		merges, _ := parseMergeArray(rawBytes)
		data.Merges = merges

	case "tokenizer.ggml.bos_token_id":
		if len(rawBytes) >= 4 {
			id := binary.LittleEndian.Uint32(rawBytes[:4])
			data.BOSID = int64(id)
			data.HasBOSID = true
		}

	case "tokenizer.ggml.eos_token_id":
		if len(rawBytes) >= 4 {
			id := binary.LittleEndian.Uint32(rawBytes[:4])
			data.EOSID = int64(id)
			data.HasEOSID = true
		}

	case "tokenizer.ggml.unknown_token_id":
		if len(rawBytes) >= 4 {
			id := binary.LittleEndian.Uint32(rawBytes[:4])
			data.UNKID = int64(id)
			data.HasUNKID = true
		}

	case "tokenizer.ggml.padding_token_id":
		if len(rawBytes) >= 4 {
			id := binary.LittleEndian.Uint32(rawBytes[:4])
			data.PADID = int64(id)
			data.HasPADID = true
		}

	case "tokenizer.ggml.add_bos":
		if len(rawBytes) > 0 {
			data.AddBOS = rawBytes[0] != 0
		}

	case "tokenizer.ggml.add_eos":
		if len(rawBytes) > 0 {
			data.AddEOS = rawBytes[0] != 0
		}

	case "tokenizer.ggml.pre":
		if len(rawBytes) > 0 {
			data.PreType = mapPreType(string(rawBytes))
		}

	case "tokenizer.ggml.space_char":
		if len(rawBytes) > 0 {
			r, _ := utf8.DecodeRuneInString(string(rawBytes))
			data.SpaceChar = r
		}

	case "tokenizer.ggml.token_type":
		// Uint32 array of TokenType IDs parallel to tokenizer.ggml.tokens
		if len(rawBytes) >= 4 {
			count := uint64(len(rawBytes)) / 4
			tokenTypes := make([]int32, count)
			for i := uint64(0); i < count; i++ {
				tokenTypes[i] = int32(binary.LittleEndian.Uint32(rawBytes[i*4 : (i+1)*4]))
			}
			data.TokenType = tokenTypes
		}

	case "tokenizer.ggml.eot_token_id":
		if len(rawBytes) >= 4 {
			id := binary.LittleEndian.Uint32(rawBytes[:4])
			data.EOTID = int64(id)
			data.HasEOTID = true
		}

	case "tokenizer.ggml.eom_token_id":
		if len(rawBytes) >= 4 {
			id := binary.LittleEndian.Uint32(rawBytes[:4])
			data.EOMID = int64(id)
			data.HasEOMID = true
		}

	case "general.name":
		// Model name is informational only — not stored in TokenizerData.
	}
}

// mapGGUFArchitecture converts GGUF architecture strings to our internal model types.
func mapGGUFArchitecture(arch string) string {
	switch arch {
	case "gpt2", "phi-2", "mpt", "olmo", "jais":
		return "bpe"
	case "llama3", "falcon3", "pixtral":
		return "bpe"
	case "qwen2", "megrez":
		return "bpe"
	case "stablelm2", "hunyuan":
		return "bpe"
	case "qwen35":
		return "bpe"
	case "gpt-4o", "llama4":
		return "bpe"
	case "starcoder", "command-r", "refact":
		return "bpe"
	case "deepseek-llm", "deepseek-coder":
		return "bpe"
	case "falcon":
		return "spm"
	case "gemma", "palm":
		return "unigram"
	case "bert", "roberta", "dphat":
		return "wordpiece"
	default:
		if strings.Contains(arch, "bpe") {
			return "bpe"
		}
		if strings.Contains(arch, "spm") || strings.Contains(arch, "sentencepiece") {
			return "spm"
		}
		if strings.Contains(arch, "unigram") {
			return "unigram"
		}
		if strings.Contains(arch, "wordpiece") {
			return "wordpiece"
		}
		return arch // fallback to raw architecture name
	}
}

// mapPreType converts GGUF pre-tokenization strategy strings to our PreType enum.
func mapPreType(pre string) common.PreType {
	switch pre {
	case "gpt-2", "phi-2", "mpt", "olmo", "jais":
		return common.PreGPT2
	case "llama3", "llama-bpe", "falcon3", "pixtral":
		return common.PreLlama3
	case "qwen2", "megrez":
		return common.PreQwen2
	case "stablelm2", "hunyuan":
		return common.PreStableLM2
	case "qwen35":
		return common.PreQwen35
	case "gpt-4o", "llama4":
		return common.PreGPT4O
	case "starcoder", "command-r", "refact":
		return common.PreStarcoder
	case "deepseek-llm", "deepseek-coder":
		return common.PreDeepSeekLLM
	case "falcon":
		return common.PreFalcon
	default:
		if strings.HasPrefix(pre, "gpt-2") {
			return common.PreGPT2
		}
		if strings.HasPrefix(pre, "llama3") || strings.Contains(pre, "llama-bpe") {
			return common.PreLlama3
		}
		if strings.HasPrefix(pre, "qwen2") {
			return common.PreQwen2
		}
		if strings.HasPrefix(pre, "starcoder") || strings.Contains(pre, "command-r") {
			return common.PreStarcoder
		}
		if strings.HasPrefix(pre, "deepseek-llm") || strings.HasPrefix(pre, "deepseek-coder") {
			return common.PreDeepSeekLLM
		}
		if strings.HasPrefix(pre, "falcon") {
			return common.PreFalcon
		}
		return common.PreDefault
	}
}

// parseStringArray parses a byte slice into a list of strings. It handles both formats:
// 1. Strings with 4-byte little-endian length prefixes per element (from readKVEntry's preserved format)
// 2. Concatenated strings without length prefixes (fallback for raw concatenated data)
func parseStringArray(raw []byte) ([]string, int) {
	if len(raw) == 0 {
		return nil, 0
	}

	// Try parsing with 4-byte length prefixes first (new format from readKVEntry)
	var tokens []string
	offset := 0
	for offset+4 <= len(raw) {
		strLen := binary.LittleEndian.Uint32(raw[offset : offset+4])
		if int(strLen)+offset+4 > len(raw) {
			break // No more complete strings
		}
		tokens = append(tokens, string(raw[offset+4:offset+4+int(strLen)]))
		offset += 4 + int(strLen)
	}

	if offset == len(raw) && len(tokens) > 0 {
		return tokens, len(tokens)
	}

	// Fallback: try parsing concatenated strings without length prefixes (old format)
	tokens = nil
	offset = 0
	for offset < len(raw) {
		// Find string boundary - look for printable ASCII or UTF-8 sequence
		end := offset + 1
		for end < len(raw) && raw[end] != 0 && end-offset < 256 {
			end++
		}
		if end == offset+1 || end-offset > 200 { // empty or too long (likely wrong parse)
			break
		}
		tokens = append(tokens, string(raw[offset:end]))
		offset = end
	}

	if len(tokens) > 0 && offset == len(raw) {
		return tokens, len(tokens)
	}

	return nil, 0
}

// parseMergeArray parses a byte slice into a list of Merge structs. Same dual-format support as parseStringArray.
func parseMergeArray(raw []byte) ([]common.Merge, int) {
	tokens, count := parseStringArray(raw)
	if tokens == nil {
		return nil, 0
	}

	merges := make([]common.Merge, 0, count)
	for _, tok := range tokens {
		parts := strings.SplitN(tok, " ", 2)
		if len(parts) == 2 {
			merges = append(merges, common.Merge{A: parts[0], B: parts[1]})
		} else if len(parts) == 1 {
			merges = append(merges, common.Merge{A: parts[0]})
		}
	}

	return merges, len(merges)
}
