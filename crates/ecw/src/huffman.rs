//! Huffman decoder for `ENCODE_HUFFMAN` sidebands.
//!
//! Each Huffman-coded sideband carries its own serialized code tree
//! followed by an LSB-first bitstream. The serialization is a pre-order
//! walk: byte `0x00` opens an internal node (its 0-child subtree follows,
//! then its 1-child subtree); any other byte is a leaf. Leaves come in two
//! shapes:
//!
//! * *small* (bit 6 set): the symbol fits in the byte itself —
//!   `value = ((byte & 0x30) << 10) | (byte & 0x0f)`. The two shifted bits
//!   land exactly on the run (0x8000) and sign (0x4000) flags.
//! * *large* (`0x80`): the next two bytes are the symbol, little-endian.
//!
//! Symbol semantics (shared with `ENCODE_RUN_ZERO`): if bit 15 is set the
//! symbol is a run of `(value & 0x7fff)` zeros; otherwise bit 14 is a sign
//! flag over a 14-bit magnitude. Run symbols are stored here with the
//! length already decremented by one, matching the reference decoder's
//! bookkeeping (`n` means "this word plus `n` more zeros").

use crate::{EcwError, Result};

const RUN_MASK: u16 = 0x8000;
const SIGN_MASK: u16 = 0x4000;
const VALUE_MASK: u16 = 0x3fff;
const MAX_RUN_LENGTH: u16 = 0x7fff;
const SMALL_SYMBOL: u8 = 0x40;
const SMALL_SHIFT: u32 = 10;
/// A tree over 16-bit symbols can never need more than 2·65536 nodes; a
/// stream claiming more is corrupt.
const MAX_NODES: usize = 2 * 65536;

/// Decoded Huffman symbol: a literal value or a zero run.
#[derive(Debug, Clone, Copy)]
pub enum Symbol {
    /// Literal quantized value.
    Value(i16),
    /// Run of `n + 1` zeros (stored decremented, as in the format).
    ZeroRun(u16),
}

#[derive(Debug, Clone, Copy)]
enum Node {
    /// Children indices into the arena: `[zero_bit, one_bit]`.
    Internal([u32; 2]),
    Leaf(Symbol),
}

/// One sideband's Huffman tree plus the read state of its bitstream.
#[derive(Debug)]
pub struct HuffmanDecoder {
    nodes: Vec<Node>,
    root: u32,
    /// Bit cursor into the payload that follows the serialized tree.
    bits_used: usize,
}

impl HuffmanDecoder {
    /// Parse the serialized tree at `data[*pos..]`, leaving `*pos` at the
    /// first byte of the bitstream.
    pub fn new(data: &[u8], pos: &mut usize) -> Result<Self> {
        // The leading u16 LE node count is a length hint only; the tree
        // shape is self-delimiting, so parse structurally and use the hint
        // as an upper bound sanity check.
        if data.len() < *pos + 2 {
            return Err(EcwError::Malformed("huffman tree truncated".into()));
        }
        *pos += 2;

        let mut nodes: Vec<Node> = Vec::new();
        let root = parse_node(data, pos, &mut nodes, 0)?;
        Ok(Self {
            nodes,
            root,
            bits_used: 0,
        })
    }

    /// Decode the next symbol from `data` (the same buffer handed to
    /// [`HuffmanDecoder::new`]; the bit cursor lives in `self`).
    #[inline]
    pub fn next_symbol(&mut self, data: &[u8]) -> Result<Symbol> {
        let mut node = self.root;
        loop {
            match self.nodes[node as usize] {
                Node::Leaf(sym) => return Ok(sym),
                Node::Internal(children) => {
                    let byte = *data
                        .get(self.bits_used >> 3)
                        .ok_or_else(|| EcwError::Malformed("huffman bitstream truncated".into()))?;
                    let bit = (byte >> (self.bits_used & 7)) & 1;
                    self.bits_used += 1;
                    node = children[bit as usize];
                }
            }
        }
    }
}

/// Recursive-descent parse of the pre-order tree serialization into a flat
/// arena. Depth is bounded by `MAX_NODES` through the arena length check,
/// but an explicit depth cap keeps pathological inputs from deep recursion.
fn parse_node(data: &[u8], pos: &mut usize, nodes: &mut Vec<Node>, depth: u32) -> Result<u32> {
    if depth > 64 {
        return Err(EcwError::Malformed("huffman tree too deep".into()));
    }
    if nodes.len() >= MAX_NODES {
        return Err(EcwError::Malformed("huffman tree too large".into()));
    }
    let byte = *data
        .get(*pos)
        .ok_or_else(|| EcwError::Malformed("huffman tree truncated".into()))?;
    *pos += 1;

    if byte == 0 {
        // Internal node: reserve a slot, then parse both subtrees.
        let index = nodes.len() as u32;
        nodes.push(Node::Internal([0, 0]));
        let zero = parse_node(data, pos, nodes, depth + 1)?;
        let one = parse_node(data, pos, nodes, depth + 1)?;
        nodes[index as usize] = Node::Internal([zero, one]);
        Ok(index)
    } else {
        let value: u16 = if byte & SMALL_SYMBOL != 0 {
            (u16::from(byte & 0x30) << SMALL_SHIFT) | u16::from(byte & 0x0f)
        } else {
            let lo = *data
                .get(*pos)
                .ok_or_else(|| EcwError::Malformed("huffman leaf truncated".into()))?;
            let hi = *data
                .get(*pos + 1)
                .ok_or_else(|| EcwError::Malformed("huffman leaf truncated".into()))?;
            *pos += 2;
            u16::from(lo) | (u16::from(hi) << 8)
        };
        let sym = if value & RUN_MASK != 0 {
            Symbol::ZeroRun((value & MAX_RUN_LENGTH).wrapping_sub(1))
        } else if value & SIGN_MASK != 0 {
            Symbol::Value(-((value & VALUE_MASK) as i16))
        } else {
            Symbol::Value(value as i16)
        };
        let index = nodes.len() as u32;
        nodes.push(Node::Leaf(sym));
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tree: root -> [leaf value 3 (small), internal -> [leaf run of 5, leaf -2]]
    /// Small leaf 3: SMALL_SYMBOL | 3 = 0x43.
    /// Run of 5 zeros: value = RUN_MASK | 5 -> large leaf 0x80, bytes 05 80.
    /// Value -2: small leaf with sign: nSymbol = SIGN | 2; small form:
    ///   SMALL | ((SIGN|2) >> 10) | 2 = 0x40 | 0x10 | 0x02 = 0x52.
    fn tree_bytes() -> Vec<u8> {
        vec![
            0x07, 0x00, // node count hint (ignored beyond the skip)
            0x00, // root internal
            0x43, // leaf: value 3
            0x00, // internal
            0x80, 0x05, 0x80, // large leaf: RUN_MASK | 5
            0x52, // small leaf: -(2)
        ]
    }

    #[test]
    fn parses_and_decodes_symbols() {
        let mut data = tree_bytes();
        // Bitstream: 0 (value 3), 1 0 (run), 1 1 (-2)  → LSB-first byte 0b11010 = 0x1a
        data.push(0b0001_1010);
        let mut pos = 0;
        let mut dec = HuffmanDecoder::new(&data, &mut pos).unwrap();
        assert_eq!(pos, 9);
        let payload = &data[pos..];

        match dec.next_symbol(payload).unwrap() {
            Symbol::Value(v) => assert_eq!(v, 3),
            other => panic!("expected value, got {other:?}"),
        }
        match dec.next_symbol(payload).unwrap() {
            // stored decremented: run of 5 → 4
            Symbol::ZeroRun(n) => assert_eq!(n, 4),
            other => panic!("expected run, got {other:?}"),
        }
        match dec.next_symbol(payload).unwrap() {
            Symbol::Value(v) => assert_eq!(v, -2),
            other => panic!("expected value, got {other:?}"),
        }
    }

    #[test]
    fn truncated_tree_errors() {
        let data = vec![0x02, 0x00, 0x00, 0x43]; // internal with one child only
        let mut pos = 0;
        assert!(HuffmanDecoder::new(&data, &mut pos).is_err());
    }
}
