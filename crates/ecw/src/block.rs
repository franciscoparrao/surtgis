//! Data-block parsing and per-sideband stream decoding.
//!
//! On disk a block holds, for every band, one entry per sideband stored at
//! this level (4 at level 0 — LL, LH, HL, HH — and 3 above it, where the LL
//! comes from reconstructing the smaller level). The layout is:
//!
//! ```text
//! u32 × (n_sidebands_total − 1)   big-endian cumulative offsets of
//!                                 sidebands 1.. relative to the data area
//! data area:
//!   per sideband (band-major): u8 encode format, then the payload
//! ```
//!
//! Every sideband is an independent stream of `block_width × block_height`
//! quantized i16 values in one of six encodings. Decoders here expose a
//! uniform sequential interface — [`Block::skip_values`] /
//! [`Block::read_values`] — so the reconstruction engine can consume any
//! encoding row by row without knowing which one it is.

use crate::huffman::{HuffmanDecoder, Symbol};
use crate::range::{QsModel, RangeDecoder, decode_symbol};
use crate::{EcwError, Result};
use std::ops::Range;

const RUN_MASK: u16 = 0x8000;
const SIGN_MASK: u16 = 0x4000;
const VALUE_MASK: u16 = 0x3fff;
const MAX_RUN_LENGTH: u16 = 0x7fff;

/// Encode formats a sideband payload can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeFormat {
    /// Verbatim little-endian i16 values.
    Raw,
    /// Zero-run + Huffman coded values.
    Huffman,
    /// Range coded i16 values (two model symbols per value).
    Range,
    /// Range coded i8 differences over a 16-bit start value.
    Range8,
    /// The whole sideband is zero.
    Zeros,
    /// u16 stream with embedded zero runs.
    RunZero,
}

impl EncodeFormat {
    fn from_raw(v: u8) -> Result<Self> {
        Ok(match v {
            1 => EncodeFormat::Raw,
            2 => EncodeFormat::Huffman,
            3 => EncodeFormat::Range,
            4 => EncodeFormat::Range8,
            5 => EncodeFormat::Zeros,
            6 => EncodeFormat::RunZero,
            other => {
                return Err(EcwError::Malformed(format!(
                    "invalid sideband encode format {other}"
                )));
            }
        })
    }
}

#[derive(Debug)]
enum DecoderState {
    Zeros,
    Raw {
        /// Position in i16 units within the payload.
        pos: usize,
    },
    RunZero {
        pos: usize,
        pending_zeros: u32,
    },
    Huffman {
        dec: HuffmanDecoder,
        /// Start of the bitstream (the tree prefix has been consumed).
        payload_from: usize,
        pending_zeros: u32,
    },
    Range {
        rc: RangeDecoder,
        model: QsModel,
    },
    Range8 {
        rc: RangeDecoder,
        model: QsModel,
        prev: i16,
    },
}

#[derive(Debug)]
struct SidebandStream {
    payload: Range<usize>,
    state: DecoderState,
}

/// One parsed data block with its per-band, per-sideband decoder states.
#[derive(Debug)]
pub struct Block {
    data: Vec<u8>,
    /// Band-major: `streams[band * sidebands_per_block + sideband_slot]`.
    streams: Vec<SidebandStream>,
}

impl Block {
    /// Parse a raw block. `sidebands` is 4 at level 0 and 3 above it;
    /// an empty `data` (a zero-length block in the offset table) yields an
    /// all-zeros block, matching the reference decoder's fallback.
    pub fn parse(data: Vec<u8>, nr_bands: usize, sidebands: usize) -> Result<Self> {
        let n = nr_bands * sidebands;
        if data.is_empty() {
            let streams = (0..n)
                .map(|_| SidebandStream {
                    payload: 0..0,
                    state: DecoderState::Zeros,
                })
                .collect();
            return Ok(Self { data, streams });
        }

        let table_len = 4 * (n - 1);
        if data.len() < table_len + n {
            return Err(EcwError::Malformed("block shorter than its header".into()));
        }
        // Cumulative big-endian offsets of sidebands 1.. within the data
        // area; sideband 0 starts at offset 0.
        let mut starts = Vec::with_capacity(n + 1);
        starts.push(0usize);
        for i in 0..n - 1 {
            let b = &data[4 * i..4 * i + 4];
            starts.push(u32::from_be_bytes(b.try_into().unwrap()) as usize);
        }
        let area = table_len;
        let area_len = data.len() - area;
        starts.push(area_len);
        if starts.windows(2).any(|w| w[1] < w[0]) || starts.iter().any(|&s| s > area_len) {
            return Err(EcwError::Malformed("sideband offsets out of range".into()));
        }

        let mut streams = Vec::with_capacity(n);
        for i in 0..n {
            let begin = area + starts[i];
            let end = area + starts[i + 1];
            if begin >= end {
                return Err(EcwError::Malformed("empty sideband entry".into()));
            }
            let format = EncodeFormat::from_raw(data[begin])?;
            let payload = begin + 1..end;
            let state = match format {
                EncodeFormat::Zeros => DecoderState::Zeros,
                EncodeFormat::Raw => DecoderState::Raw { pos: 0 },
                EncodeFormat::RunZero => DecoderState::RunZero {
                    pos: 0,
                    pending_zeros: 0,
                },
                EncodeFormat::Huffman => {
                    let mut pos = payload.start;
                    let dec = HuffmanDecoder::new(&data, &mut pos)?;
                    DecoderState::Huffman {
                        dec,
                        payload_from: pos,
                        pending_zeros: 0,
                    }
                }
                EncodeFormat::Range => {
                    let mut pos = payload.start;
                    let rc = RangeDecoder::new(&data, &mut pos);
                    DecoderState::Range {
                        rc,
                        model: QsModel::new(),
                    }
                }
                EncodeFormat::Range8 => {
                    let mut pos = payload.start;
                    let mut rc = RangeDecoder::new(&data, &mut pos);
                    let mut model = QsModel::new();
                    // The stream opens with one 16-bit value, high byte
                    // first, that seeds the difference accumulator.
                    let hi = decode_symbol(&mut rc, &mut model, &data);
                    let lo = decode_symbol(&mut rc, &mut model, &data);
                    let prev = (((hi as u16 & 0xff) << 8) | (lo as u16 & 0xff)) as i16;
                    DecoderState::Range8 { rc, model, prev }
                }
            };
            streams.push(SidebandStream { payload, state });
        }
        Ok(Self { data, streams })
    }

    /// Skip `count` values of stream `index` (band-major sideband index).
    pub fn skip_values(&mut self, index: usize, count: usize) -> Result<()> {
        let data = &self.data;
        let stream = &mut self.streams[index];
        let payload = &data[stream.payload.clone()];
        match &mut stream.state {
            DecoderState::Zeros => {}
            DecoderState::Raw { pos } => *pos += count,
            DecoderState::RunZero { pos, pending_zeros } => {
                let mut left = count;
                while left > 0 {
                    if *pending_zeros > 0 {
                        let take = (*pending_zeros as usize).min(left);
                        *pending_zeros -= take as u32;
                        left -= take;
                        continue;
                    }
                    let v = read_u16_le(payload, pos)?;
                    if v & RUN_MASK != 0 {
                        *pending_zeros = u32::from(v & MAX_RUN_LENGTH);
                    } else {
                        left -= 1;
                    }
                }
            }
            DecoderState::Huffman {
                dec,
                payload_from,
                pending_zeros,
            } => {
                let bits = &data[*payload_from..stream.payload.end];
                let mut left = count;
                while left > 0 {
                    if *pending_zeros > 0 {
                        let take = (*pending_zeros as usize).min(left);
                        *pending_zeros -= take as u32;
                        left -= take;
                        continue;
                    }
                    match dec.next_symbol(bits)? {
                        Symbol::ZeroRun(n) => *pending_zeros = u32::from(n) + 1,
                        Symbol::Value(_) => left -= 1,
                    }
                }
            }
            DecoderState::Range { rc, model } => {
                for _ in 0..count * 2 {
                    decode_symbol(rc, model, data);
                }
            }
            DecoderState::Range8 { rc, model, prev } => {
                for _ in 0..count {
                    let ch = decode_symbol(rc, model, data);
                    *prev = prev.wrapping_add((ch as u8 as i8) as i16);
                }
            }
        }
        Ok(())
    }

    /// Decode `out.len()` values of stream `index`, dequantizing with
    /// `bin_size` into f32.
    pub fn read_values(&mut self, index: usize, out: &mut [f32], bin_size: f32) -> Result<()> {
        let data = &self.data;
        let stream = &mut self.streams[index];
        let payload = &data[stream.payload.clone()];
        match &mut stream.state {
            DecoderState::Zeros => out.fill(0.0),
            DecoderState::Raw { pos } => {
                for slot in out.iter_mut() {
                    let base = *pos * 2;
                    let lo = *payload.get(base).ok_or_else(raw_truncated)?;
                    let hi = *payload.get(base + 1).ok_or_else(raw_truncated)?;
                    *slot = f32::from(i16::from_le_bytes([lo, hi])) * bin_size;
                    *pos += 1;
                }
            }
            DecoderState::RunZero { pos, pending_zeros } => {
                let mut i = 0;
                while i < out.len() {
                    if *pending_zeros > 0 {
                        let take = (*pending_zeros as usize).min(out.len() - i);
                        out[i..i + take].fill(0.0);
                        *pending_zeros -= take as u32;
                        i += take;
                        continue;
                    }
                    let v = read_u16_le(payload, pos)?;
                    if v & RUN_MASK != 0 {
                        *pending_zeros = u32::from(v & MAX_RUN_LENGTH);
                    } else if v & SIGN_MASK != 0 {
                        out[i] = -f32::from(v & VALUE_MASK) * bin_size;
                        i += 1;
                    } else {
                        out[i] = f32::from(v) * bin_size;
                        i += 1;
                    }
                }
            }
            DecoderState::Huffman {
                dec,
                payload_from,
                pending_zeros,
            } => {
                let bits = &data[*payload_from..stream.payload.end];
                let mut i = 0;
                while i < out.len() {
                    if *pending_zeros > 0 {
                        let take = (*pending_zeros as usize).min(out.len() - i);
                        out[i..i + take].fill(0.0);
                        *pending_zeros -= take as u32;
                        i += take;
                        continue;
                    }
                    match dec.next_symbol(bits)? {
                        Symbol::ZeroRun(n) => *pending_zeros = u32::from(n) + 1,
                        Symbol::Value(v) => {
                            out[i] = f32::from(v) * bin_size;
                            i += 1;
                        }
                    }
                }
            }
            DecoderState::Range { rc, model } => {
                for slot in out.iter_mut() {
                    let lo = decode_symbol(rc, model, data);
                    if lo == 256 {
                        return Err(EcwError::Malformed(
                            "unexpected end marker in range stream".into(),
                        ));
                    }
                    let hi = decode_symbol(rc, model, data);
                    let value = ((lo as u16) & 0xff | (((hi as u16) & 0xff) << 8)) as i16;
                    *slot = f32::from(value) * bin_size;
                }
            }
            DecoderState::Range8 { rc, model, prev } => {
                for slot in out.iter_mut() {
                    let ch = decode_symbol(rc, model, data);
                    if ch == 256 {
                        return Err(EcwError::Malformed(
                            "unexpected end marker in range8 stream".into(),
                        ));
                    }
                    *prev = prev.wrapping_add((ch as u8 as i8) as i16);
                    *slot = f32::from(*prev) * bin_size;
                }
            }
        }
        Ok(())
    }
}

fn raw_truncated() -> EcwError {
    EcwError::Malformed("raw sideband truncated".into())
}

fn read_u16_le(payload: &[u8], pos: &mut usize) -> Result<u16> {
    let base = *pos * 2;
    let lo = *payload
        .get(base)
        .ok_or_else(|| EcwError::Malformed("run-zero sideband truncated".into()))?;
    let hi = *payload
        .get(base + 1)
        .ok_or_else(|| EcwError::Malformed("run-zero sideband truncated".into()))?;
    *pos += 1;
    Ok(u16::from(lo) | (u16::from(hi) << 8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-band level-0 style block with 4 RAW sidebands.
    fn raw_block(values: [i16; 4]) -> Vec<u8> {
        let mut area = Vec::new();
        let mut offsets = Vec::new();
        for (i, v) in values.iter().enumerate() {
            if i > 0 {
                offsets.push(area.len() as u32);
            }
            area.push(1u8); // ENCODE_RAW
            area.extend_from_slice(&v.to_le_bytes());
        }
        // offsets are of sidebands 1..: recompute as cumulative starts
        let mut data = Vec::new();
        for (i, _) in values.iter().enumerate().skip(1) {
            data.extend_from_slice(&((i * 3) as u32).to_be_bytes());
        }
        data.extend_from_slice(&area);
        data
    }

    #[test]
    fn parses_raw_sidebands() {
        let data = raw_block([10, -3, 0, 7]);
        let mut block = Block::parse(data, 1, 4).unwrap();
        let mut out = [0.0f32; 1];
        for (i, expected) in [10.0, -3.0, 0.0, 7.0].into_iter().enumerate() {
            block.read_values(i, &mut out, 1.0).unwrap();
            assert_eq!(out[0], expected);
        }
    }

    #[test]
    fn run_zero_roundtrip() {
        // Payload: value 5, run of 3 zeros, value -2 (sign bit).
        let mut area = vec![6u8]; // ENCODE_RUN_ZERO
        for v in [5u16, RUN_MASK | 3, SIGN_MASK | 2] {
            area.extend_from_slice(&v.to_le_bytes());
        }
        let block_data = area; // single band, single sideband → no offset table
        let mut block = Block::parse(block_data, 1, 1).unwrap();
        let mut out = [9.0f32; 5];
        block.read_values(0, &mut out, 2.0).unwrap();
        assert_eq!(out, [10.0, 0.0, 0.0, 0.0, -4.0]);
    }

    #[test]
    fn run_zero_skip_preserves_pending() {
        // Run of 4 zeros then value 1: skipping 2 must leave 2 pending.
        let mut area = vec![6u8];
        for v in [RUN_MASK | 4, 1u16] {
            area.extend_from_slice(&v.to_le_bytes());
        }
        let mut block = Block::parse(area, 1, 1).unwrap();
        block.skip_values(0, 2).unwrap();
        let mut out = [7.0f32; 3];
        block.read_values(0, &mut out, 1.0).unwrap();
        assert_eq!(out, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn empty_block_is_all_zeros() {
        let mut block = Block::parse(Vec::new(), 3, 4).unwrap();
        let mut out = [5.0f32; 8];
        block.read_values(11, &mut out, 1.0).unwrap();
        assert_eq!(out, [0.0; 8]);
    }
}
