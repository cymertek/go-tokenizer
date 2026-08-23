package spm

// Minimal protobuf decoder for SentencePiece model binaries.
// Only decodes the fields we need: ModelProto.Pieces[].ID, .Word, .Score.

import (
	"encoding/binary"
	"fmt"
)

// protoField represents a single protobuf field during decoding.
type protoField struct {
	tag    uint64
	length int
	data   []byte
}

// decodeVarint reads a varint from the byte slice starting at pos.
func decodeVarint(data []byte, pos int) (uint64, int, error) {
	var result uint64
	var shift uint
	for pos < len(data) {
		b := data[pos]
		pos++
		result |= uint64(b&0x7F) << shift
		if b&0x80 == 0 {
			return result, pos, nil
		}
		shift += 7
		if shift > 63 {
			return 0, 0, fmt.Errorf("varint too long")
		}
	}
	return 0, 0, fmt.Errorf("unexpected end of varint")
}

// decodeLengthDelimited reads a length-prefixed byte slice.
func decodeLengthDelimited(data []byte, pos int) ([]byte, int, error) {
	length, newPos, err := decodeVarint(data, pos)
	if err != nil {
		return nil, 0, err
	}
	endPos := newPos + int(length)
	if endPos > len(data) {
		return nil, 0, fmt.Errorf("length-delimited field extends beyond data")
	}
	return data[newPos:endPos], endPos, nil
}

// piece represents a single SentencePiece vocabulary entry.
type piece struct {
	ID    uint32
	Word  string
	Score float32
}

// decodeModelProto parses the protobuf-encoded model binary and extracts pieces.
func decodeModelProto(data []byte) ([]piece, error) {
	var pieces []piece
	pos := 0

	for pos < len(data) {
		tag, newPos, err := decodeVarint(data, pos)
		if err != nil {
			return nil, fmt.Errorf("decode tag at %d: %w", pos, err)
		}
		fieldNumber := tag >> 3
		wireType := tag & 0x7

		switch fieldNumber {
		case 12: // pieces (repeated MessageField)
			if wireType != 2 {
				return nil, fmt.Errorf("expected wire type 2 for pieces, got %d", wireType)
			}
			pieceData, endPos, err := decodeLengthDelimited(data, newPos)
			if err != nil {
				return nil, fmt.Errorf("decode piece message: %w", err)
			}
			p, err := decodePieceMessage(pieceData)
			if err == nil && p.ID != 0 {
				pieces = append(pieces, *p)
			}
			pos = endPos

		case 16: // unk_id (uint32) — skip for now
			if wireType != 0 {
				return nil, fmt.Errorf("expected wire type 0 for unk_id")
			}
			val, _, err := decodeVarint(data, newPos)
			if err == nil && val > 0 {
				// Could store this but not needed for basic functionality
			}
			pos = newPos

		default:
			// Skip unknown fields based on wire type
			var skipEnd int
			switch wireType {
			case 0: // varint
				_, newPos, err := decodeVarint(data, pos)
				if err != nil {
					return nil, err
				}
				skipEnd = newPos
			case 2: // length-delimited
				dataBytes, endPos, err := decodeLengthDelimited(data, pos)
				if err != nil {
					return nil, err
				}
				_ = dataBytes
				skipEnd = endPos
			default:
				return nil, fmt.Errorf("unsupported wire type %d at field %d", wireType, fieldNumber)
			}
			pos = skipEnd
		}
	}

	// Sort pieces by ID to ensure correct ordering
	for i := 0; i < len(pieces); i++ {
		for j := i + 1; j < len(pieces); j++ {
			if pieces[j].ID < pieces[i].ID {
				pieces[i], pieces[j] = pieces[j], pieces[i]
			}
		}
	}

	return pieces, nil
}

// decodePieceMessage parses a single piece message from protobuf bytes.
func decodePieceMessage(data []byte) (*piece, error) {
	p := &piece{}
	pos := 0

	for pos < len(data) {
		tag, newPos, err := decodeVarint(data, pos)
		if err != nil {
			return nil, err
		}
		fieldNumber := tag >> 3
		wireType := tag & 0x7

		switch fieldNumber {
		case 1: // id (uint32)
			if wireType != 0 {
				return nil, fmt.Errorf("expected wire type 0 for id")
			}
			val, _, err := decodeVarint(data, newPos)
			if err == nil {
				p.ID = uint32(val)
			}

		case 2: // piece (string)
			if wireType != 2 {
				return nil, fmt.Errorf("expected wire type 2 for piece")
			}
			s, endPos, err := decodeLengthDelimited(data, newPos)
			if err == nil {
				p.Word = string(s)
			}
			pos = endPos
			continue

		case 3: // score (float)
			if wireType != 5 { // fixed32
				return nil, fmt.Errorf("expected wire type 5 for score")
			}
			if newPos+4 > len(data) {
				return nil, fmt.Errorf("score field too short")
			}
			p.Score = float32(binary.LittleEndian.Uint32(data[newPos : newPos+4]))

		default:
			// Skip unknown fields — track previous position to detect infinite loops.
			prevPos := pos
			var skipEnd int
			switch wireType {
			case 0:
				_, newPos, err := decodeVarint(data, prevPos)
				if err != nil {
					return nil, err
				}
				skipEnd = newPos
			case 2:
				_, endPos, err := decodeLengthDelimited(data, prevPos)
				if err != nil {
					return nil, err
				}
				skipEnd = endPos
			default:
				return nil, fmt.Errorf("unsupported wire type %d in piece", wireType)
			}
			pos = skipEnd
		}

		if pos == newPos {
			break // avoid infinite loop
		}
		pos = newPos
	}

	return p, nil
}
