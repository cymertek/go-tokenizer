// Package proto provides binary serialization for tokenizers in a custom protobuf-like format.
// This format captures all state needed for full-fidelity round-tripping across BPE, SPM,
// Unigram, and WordPiece tokenizer formats — including vocabularies, merge rules, probabilities,
// special tokens, normalizer/pretokenizer/decoder configuration, and arbitrary config maps.
//
// Wire format:
//   - Magic header: "TOKD" (4 bytes)
//   - Version byte: 2 (current version)
//   - Length-prefixed key-value fields until EOF
//
// Field keys:
//   'M' Model string, 'T' Tokens ([]string null-terminated),
//   'B' BOSID (+Has flag), 'E' EOSID (+Has flag), 'F' EOTID (+Has flag), 'H' EOMID (+Has flag),
//   'U' UNKID (+Has flag), 'P' PADID (+Has flag),
//   'A' AddBOS bool, 'S' AddEOS bool,
//   'Y' PreType (int32), 'C' SpaceChar rune (uint32),
//   'm' Merge rule (single lowercase key to avoid conflict with uppercase 'M'),
//   'V' SPMProbabilities ([]float64 interleaved id/score pairs),
//   'W' TokenType ([]int32 parallel to Tokens),
//   'N' NormalizerConfig (JSON blob),
//   'R' PreTokenizerConfig (JSON blob),
//   'D' DecoderConfig (JSON blob),
//   'X' SPMModel binary,
//   'Z' ConfigMap entry (serialized as JSON).
package proto

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/cymertek/go-tokenizer/models/common"
)

// TokenizerData holds all tokenizer configuration fields — mirrors common.TokenizerData.
type TokenizerData = common.TokenizerData

// Merge represents a single BPE merge rule — mirrors common.Merge.
type Merge = common.Merge

// NormalizerConfig holds serialized normalizer state.
type NormalizerConfig struct {
	Type   string `json:"type"`
	Config []byte `json:"config,omitempty"` // JSON-serialized parameters
}

// PreTokenizerConfig holds serialized pre-tokenizer state.
type PreTokenizerConfig struct {
	Type   string `json:"type"`
	Config []byte `json:"config,omitempty"` // JSON-serialized parameters
}

// DecoderConfig holds serialized decoder state.
type DecoderConfig struct {
	Type   string `json:"type"`
	Config []byte `json:"config,omitempty"` // JSON-serialized parameters
}

// ConfigEntry represents a single key-value pair in the config map.
type ConfigEntry struct {
	Key   string      `json:"key"`
	Value interface{} `json:"value"`
}

// Tokenizer is the complete, serializable tokenizer with all components.
type Tokenizer struct {
	Data       *TokenizerData        `json:"data,omitempty"`
	Normalizer *NormalizerConfig     `json:"normalizer,omitempty"`
	PreTok     *PreTokenizerConfig   `json:"pre_toker,omitempty"`
	Decoder    *DecoderConfig        `json:"decoder,omitempty"`
	ConfigMap  []ConfigEntry         `json:"config_map,omitempty"`
}

// Serialize writes the Tokenizer to w in binary protobuf format.
func (t *Tokenizer) Serialize(w io.Writer) error {
	if t == nil || t.Data == nil {
		return fmt.Errorf("Serialize: tokenizer or data is nil")
	}

	// Write magic header "TOKD"
	header := []byte{'T', 'O', 'K', 'D'}
	if _, err := w.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	// Write version (1 byte) — version 2 supports all four formats fully
	version := uint8(2)
	if err := binary.Write(w, binary.LittleEndian, version); err != nil {
		return fmt.Errorf("write version: %w", err)
	}

	// Serialize TokenizerData fields as length-prefixed strings/values.
	// Only write keys for fields that have non-nil values to avoid format mismatch.
	writeField := func(key byte, value interface{}) error {
		data, ok := serializeValue(value)
		if !ok {
			return nil // skip unset fields entirely (no key written)
		}
		if err := binary.Write(w, binary.LittleEndian, key); err != nil {
			return fmt.Errorf("write field key: %w", err)
		}
		return writeLengthPrefixed(w, data)
	}

	if err := writeField('M', t.Data.Model); err != nil {
		return err
	}
	if err := writeField('T', serializeStringSlice(t.Data.Tokens)); err != nil {
		return err
	}
	if err := writeField('B', serializeInt64WithFlag(t.Data.BOSID, t.Data.HasBOSID)); err != nil {
		return err
	}
	if err := writeField('E', serializeInt64WithFlag(t.Data.EOSID, t.Data.HasEOSID)); err != nil {
		return err
	}
	if err := writeField('F', serializeInt64WithFlag(t.Data.EOTID, t.Data.HasEOTID)); err != nil {
		return err
	}
	if err := writeField('H', serializeInt64WithFlag(t.Data.EOMID, t.Data.HasEOMID)); err != nil {
		return err
	}
	if err := writeField('U', serializeInt64WithFlag(t.Data.UNKID, t.Data.HasUNKID)); err != nil {
		return err
	}
	if err := writeField('P', serializeInt64WithFlag(t.Data.PADID, t.Data.HasPADID)); err != nil {
		return err
	}
	if err := writeField('A', boolByte(t.Data.AddBOS)); err != nil {
		return err
	}
	if err := writeField('S', boolByte(t.Data.AddEOS)); err != nil {
		return err
	}
	if err := writeField('Y', int32ToBytes(int32(t.Data.PreType))); err != nil {
		return err
	}
	if err := writeField('C', uint32ToBytes(uint32(t.Data.SpaceChar))); err != nil {
		return err
	}

	// Serialize merges if present (single lowercase key 'm')
	if len(t.Data.Merges) > 0 {
		for _, m := range t.Data.Merges {
			if err := binary.Write(w, binary.LittleEndian, byte('m')); err != nil {
				return fmt.Errorf("write merge key: %w", err)
			}
			mergeData := serializeMerge(m)
			if err := writeLengthPrefixed(w, mergeData); err != nil {
				return fmt.Errorf("write merge data: %w", err)
			}
		}
	}

	// Serialize SPM probabilities (interleaved [id, score] float64 pairs) for Unigram Viterbi
	if len(t.Data.SPMProbabilities) > 0 {
		if err := binary.Write(w, binary.LittleEndian, byte('V')); err != nil {
			return fmt.Errorf("write spm prob key: %w", err)
		}
		probBytes := serializeFloat64Slice(t.Data.SPMProbabilities)
		if err := writeLengthPrefixed(w, probBytes); err != nil {
			return fmt.Errorf("write spm prob data: %w", err)
		}
	}

	// Serialize TokenType array (parallel to Tokens)
	if len(t.Data.TokenType) > 0 {
		if err := binary.Write(w, binary.LittleEndian, byte('W')); err != nil {
			return fmt.Errorf("write token type key: %w", err)
		}
		typeBytes := serializeInt32Slice(t.Data.TokenType)
		if err := writeLengthPrefixed(w, typeBytes); err != nil {
			return fmt.Errorf("write token type data: %w", err)
		}
	}

	// Serialize SPM model binary (for SentencePiece tokenizers from GGUF files)
	if len(t.Data.SPMModel) > 0 {
		if err := binary.Write(w, binary.LittleEndian, byte('X')); err != nil {
			return fmt.Errorf("write spm key: %w", err)
		}
		if err := writeLengthPrefixed(w, t.Data.SPMModel); err != nil {
			return fmt.Errorf("write spm data: %w", err)
		}
	}

	// Serialize normalizer config if present (binary format for lossless round-trip)
	if t.Normalizer != nil {
		if err := binary.Write(w, binary.LittleEndian, byte('N')); err != nil {
			return fmt.Errorf("write normalizer key: %w", err)
		}
		normBytes := serializeNormalizerConfig(t.Normalizer)
		if err := writeLengthPrefixed(w, normBytes); err != nil {
			return fmt.Errorf("write normalizer data: %w", err)
		}
	}

	// Serialize pre-tokenizer config if present (binary format for lossless round-trip)
	if t.PreTok != nil {
		if err := binary.Write(w, binary.LittleEndian, byte('R')); err != nil {
			return fmt.Errorf("write pretok key: %w", err)
		}
		pretokBytes := serializePreTokenizerConfig(t.PreTok)
		if err := writeLengthPrefixed(w, pretokBytes); err != nil {
			return fmt.Errorf("write pretok data: %w", err)
		}
	}

	// Serialize decoder config if present (binary format for lossless round-trip)
	if t.Decoder != nil {
		if err := binary.Write(w, binary.LittleEndian, byte('D')); err != nil {
			return fmt.Errorf("write decoder key: %w", err)
		}
		decBytes := serializeDecoderConfig(t.Decoder)
		if err := writeLengthPrefixed(w, decBytes); err != nil {
			return fmt.Errorf("write decoder data: %w", err)
		}
	}

	// Serialize config map entries if present (binary format for lossless round-trip)
	for _, entry := range t.ConfigMap {
		if err := binary.Write(w, binary.LittleEndian, byte('Z')); err != nil { // 'Z' = config map entry marker
			return fmt.Errorf("write config key: %w", err)
		}
		entryBytes := serializeConfigEntry(&entry)
		if err := writeLengthPrefixed(w, entryBytes); err != nil {
			return fmt.Errorf("write config data: %w", err)
		}
	}

	return nil
}

// Deserialize reads a Tokenizer from r in binary protobuf format.
func Deserialize(r io.Reader) (*Tokenizer, error) {
	t := &Tokenizer{}

	// Read and verify magic header "TOKD"
	var header [4]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(header[:]) != "TOKD" {
		return nil, fmt.Errorf("invalid magic header: got %q, want TOKD", header[:])
	}

	// Read version (1 byte)
	var version uint8
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	// Version 2 only — v1 was never released and had incomplete support
	if version != 2 {
		return nil, fmt.Errorf("unsupported version: %d (only v2 supported)", version)
	}

	t.Data = &TokenizerData{}

	// Read fields until EOF or invalid key
	for {
		var key byte
		if err := binary.Read(r, binary.LittleEndian, &key); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read field key: %w", err)
		}

		data, err := readLengthPrefixed(r)
		if err != nil {
			return nil, fmt.Errorf("read field data: %w", err)
		}

		switch key {
		case 'M':
			t.Data.Model = string(data)
		case 'T':
			t.Data.Tokens = deserializeStringSlice(data)
		case 'B':
			id, has := deserializeInt64WithFlag(data)
			if has {
				t.Data.BOSID = id
				t.Data.HasBOSID = true
			}
		case 'E':
			id, has := deserializeInt64WithFlag(data)
			if has {
				t.Data.EOSID = id
				t.Data.HasEOSID = true
			}
		case 'F': // EOTID
			id, has := deserializeInt64WithFlag(data)
			if has {
				t.Data.EOTID = id
				t.Data.HasEOTID = true
			}
		case 'H': // EOMID
			id, has := deserializeInt64WithFlag(data)
			if has {
				t.Data.EOMID = id
				t.Data.HasEOMID = true
			}
		case 'U':
			id, has := deserializeInt64WithFlag(data)
			if has {
				t.Data.UNKID = id
				t.Data.HasUNKID = true
			}
		case 'P':
			id, has := deserializeInt64WithFlag(data)
			if has {
				t.Data.PADID = id
				t.Data.HasPADID = true
			}
		case 'A':
			if len(data) >= 1 {
				t.Data.AddBOS = data[0] == 1
			}
		case 'S':
			if len(data) >= 1 {
				t.Data.AddEOS = data[0] == 1
			}
		case 'Y':
			if len(data) >= 4 {
				t.Data.PreType = common.PreType(binary.LittleEndian.Uint32(data[:4]))
			}
		case 'C':
			if len(data) >= 4 {
				t.Data.SpaceChar = rune(binary.LittleEndian.Uint32(data[:4]))
			}
		case 'V': // SPMProbabilities (interleaved [id, score] float64 pairs)
			t.Data.SPMProbabilities = deserializeFloat64Slice(data)
		case 'W': // TokenType array
			t.Data.TokenType = deserializeInt32Slice(data)
		case 'm': // Merge rule (lowercase key to avoid conflict with uppercase 'M')
			if len(data) >= 1 {
				m := deserializeMerge(data)
				if m != nil {
					t.Data.Merges = append(t.Data.Merges, *m)
				}
			}
		case 'X': // SPM model binary
			t.Data.SPMModel = data
		case 'N': // Normalizer config (binary)
			norm := deserializeNormalizerConfig(data)
			t.Normalizer = &norm
		case 'R': // PreTokenizer config (binary)
			pt := deserializePreTokenizerConfig(data)
			t.PreTok = &pt
		case 'D': // Decoder config (binary)
			dec := deserializeDecoderConfig(data)
			t.Decoder = &dec
		case 'Z': // Config map entry (binary)
			entry := deserializeConfigEntry(data)
			if entry != nil {
				t.ConfigMap = append(t.ConfigMap, *entry)
			}
		default:
			// Skip unknown fields for forward compatibility
		}
	}

	return t, nil
}

// --- Helper serialization functions ---

func serializeStringSlice(ss []string) []byte {
	if len(ss) == 0 {
		return nil
	}
	var buf []byte
	for _, s := range ss {
		buf = append(buf, []byte(s)...)
		buf = append(buf, '\x00') // null terminator
	}
	return buf
}

func deserializeStringSlice(data []byte) []string {
	if len(data) == 0 {
		return nil
	}
	var result []string
	start := 0
	for i, b := range data {
		if b == '\x00' {
			result = append(result, string(data[start:i]))
			start = i + 1
		}
	}
	return result
}

func serializeInt64WithFlag(val int64, has bool) []byte {
	if !has {
		return nil
	}
	b := make([]byte, 9) // 8 bytes for value + 1 byte flag
	binary.LittleEndian.PutUint64(b[:8], uint64(val))
	b[8] = 1
	return b
}

func deserializeInt64WithFlag(data []byte) (int64, bool) {
	if len(data) < 9 || data[8] != 1 {
		return 0, false
	}
	val := int64(binary.LittleEndian.Uint64(data[:8]))
	return val, true
}

func boolByte(b bool) []byte {
	if b {
		return []byte{1}
	}
	return nil
}

func int32ToBytes(v int32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, uint32(v))
	return b
}

func uint32ToBytes(v uint32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	return b
}

// serializeFloat64Slice serializes a slice of float64 values (used for SPMProbabilities).
func serializeFloat64Slice(vals []float64) []byte {
	if len(vals) == 0 {
		return nil
	}
	b := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[i*8:(i+1)*8], math.Float64bits(v))
	}
	return b
}

// deserializeFloat64Slice deserializes a slice of float64 values.
func deserializeFloat64Slice(data []byte) []float64 {
	if len(data) == 0 || len(data)%8 != 0 {
		return nil
	}
	n := len(data) / 8
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint64(data[i*8 : (i+1)*8])
		result[i] = math.Float64frombits(bits)
	}
	return result
}

// serializeInt32Slice serializes a slice of int32 values (used for TokenType).
func serializeInt32Slice(vals []int32) []byte {
	if len(vals) == 0 {
		return nil
	}
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:(i+1)*4], uint32(v))
	}
	return b
}

// deserializeInt32Slice deserializes a slice of int32 values.
func deserializeInt32Slice(data []byte) []int32 {
	if len(data) == 0 || len(data)%4 != 0 {
		return nil
	}
	n := len(data) / 4
	result := make([]int32, n)
	for i := 0; i < n; i++ {
		v := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		result[i] = int32(v)
	}
	return result
}

func serializeValue(v interface{}) ([]byte, bool) {
	switch val := v.(type) {
	case string:
		if val == "" {
			return nil, false
		}
		return []byte(val), true
	case []byte:
		if len(val) == 0 {
			return nil, false
		}
		return val, true
	default:
		return nil, false
	}
}

func writeLengthPrefixed(w io.Writer, data []byte) error {
	length := uint32(len(data))
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return err
	}
	_, err := w.Write(data)
	return err
}

func readLengthPrefixed(r io.Reader) ([]byte, error) {
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return nil, err
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(r, data); err != nil {
		return nil, err
	}
	return data, nil
}

func serializeMerge(m Merge) []byte {
	aBytes := []byte(m.A)
	bBytes := []byte(m.B)
	buf := make([]byte, 0, len(aBytes)+len(bBytes)+8)
	buf = append(buf, aBytes...)
	buf = append(buf, '\x00')
	buf = append(buf, bBytes...)
	return buf
}

func deserializeMerge(data []byte) *Merge {
	if len(data) == 0 {
		return nil
	}
	nullIdx := -1
	for i, b := range data {
		if b == '\x00' {
			nullIdx = i
			break
		}
	}
	if nullIdx < 0 {
		return nil
	}
	a := string(data[:nullIdx])
	b := string(data[nullIdx+1:])
	return &Merge{A: a, B: b}
}

// --- Binary serialization/deserialization for config types ---

func serializeNormalizerConfig(c *NormalizerConfig) []byte {
	typeStr := []byte(c.Type)
	configBytes := c.Config
	buf := make([]byte, 0, len(typeStr)+len(configBytes)+16)
	// Type length (4 bytes) + type string
	buf = append(buf, int32ToBytes(int32(len(typeStr)))...)
	buf = append(buf, typeStr...)
	// Config length (4 bytes) + config bytes
	buf = append(buf, int32ToBytes(int32(len(configBytes)))...)
	buf = append(buf, configBytes...)
	return buf
}

func deserializeNormalizerConfig(data []byte) NormalizerConfig {
	if len(data) < 8 {
		return NormalizerConfig{}
	}
	typeLen := binary.LittleEndian.Uint32(data[:4])
	typeStr := string(data[4 : 4+typeLen])
	configStart := int(4 + typeLen)
	if configStart+8 > len(data) {
		return NormalizerConfig{Type: typeStr}
	}
	configLen := binary.LittleEndian.Uint32(data[configStart:])
	configBytes := data[configStart+4 : configStart+4+int(configLen)]
	return NormalizerConfig{
		Type:   typeStr,
		Config: configBytes,
	}
}

func serializePreTokenizerConfig(c *PreTokenizerConfig) []byte {
	typeStr := []byte(c.Type)
	configBytes := c.Config
	buf := make([]byte, 0, len(typeStr)+len(configBytes)+16)
	buf = append(buf, int32ToBytes(int32(len(typeStr)))...)
	buf = append(buf, typeStr...)
	buf = append(buf, int32ToBytes(int32(len(configBytes)))...)
	buf = append(buf, configBytes...)
	return buf
}

func deserializePreTokenizerConfig(data []byte) PreTokenizerConfig {
	if len(data) < 8 {
		return PreTokenizerConfig{}
	}
	typeLen := binary.LittleEndian.Uint32(data[:4])
	typeStr := string(data[4 : 4+typeLen])
	configStart := int(4 + typeLen)
	if configStart+8 > len(data) {
		return PreTokenizerConfig{Type: typeStr}
	}
	configLen := binary.LittleEndian.Uint32(data[configStart:])
	configBytes := data[configStart+4 : configStart+4+int(configLen)]
	return PreTokenizerConfig{
		Type:   typeStr,
		Config: configBytes,
	}
}

func serializeDecoderConfig(c *DecoderConfig) []byte {
	typeStr := []byte(c.Type)
	configBytes := c.Config
	buf := make([]byte, 0, len(typeStr)+len(configBytes)+16)
	buf = append(buf, int32ToBytes(int32(len(typeStr)))...)
	buf = append(buf, typeStr...)
	buf = append(buf, int32ToBytes(int32(len(configBytes)))...)
	buf = append(buf, configBytes...)
	return buf
}

func deserializeDecoderConfig(data []byte) DecoderConfig {
	if len(data) < 8 {
		return DecoderConfig{}
	}
	typeLen := binary.LittleEndian.Uint32(data[:4])
	typeStr := string(data[4 : 4+typeLen])
	configStart := int(4 + typeLen)
	if configStart+8 > len(data) {
		return DecoderConfig{Type: typeStr}
	}
	configLen := binary.LittleEndian.Uint32(data[configStart:])
	configBytes := data[configStart+4 : configStart+4+int(configLen)]
	return DecoderConfig{
		Type:   typeStr,
		Config: configBytes,
	}
}

func serializeConfigEntry(e *ConfigEntry) []byte {
	keyStr := []byte(e.Key)
	valueJSON, err := json.Marshal(e.Value)
	if err != nil {
		return nil
	}
	buf := make([]byte, 0, len(keyStr)+len(valueJSON)+16)
	buf = append(buf, int32ToBytes(int32(len(keyStr)))...)
	buf = append(buf, keyStr...)
	buf = append(buf, int32ToBytes(int32(len(valueJSON)))...)
	buf = append(buf, valueJSON...)
	return buf
}

func deserializeConfigEntry(data []byte) *ConfigEntry {
	if len(data) < 8 {
		return nil
	}
	keyLen := binary.LittleEndian.Uint32(data[:4])
	keyStr := string(data[4 : 4+keyLen])
	valueStart := int(4 + keyLen)
	if valueStart+8 > len(data) {
		return &ConfigEntry{Key: keyStr, Value: nil}
	}
	valueLen := binary.LittleEndian.Uint32(data[valueStart:])
	valueJSON := data[valueStart+4 : valueStart+4+int(valueLen)]
	var value interface{}
	if err := json.Unmarshal(valueJSON, &value); err != nil {
		return &ConfigEntry{Key: keyStr, Value: string(valueJSON)}
	}
	return &ConfigEntry{Key: keyStr, Value: value}
}
