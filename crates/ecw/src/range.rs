//! Range decoder and quasistatic probability model for `ENCODE_RANGE` /
//! `ENCODE_RANGE8` sidebands.
//!
//! ECW uses byte-renormalized range coding (G.N.N. Martin, 1979) driven by
//! a 257-symbol adaptive "quasistatic" frequency model with a 12-bit total
//! (4096) and a rescale target of 2000, matching the parameters the format
//! fixes for every stream. This is a fresh implementation of the published
//! algorithm; constants below are format facts, not code lineage.
//!
//! Streams never carry an explicit length: the caller knows how many
//! values to pull. Reads past the end of the payload yield zero bytes,
//! mirroring the tolerance of the reference decoder (which reads whatever
//! bytes follow the sideband and discards the results).

/// Number of symbols in the model: 256 byte values plus one end marker.
pub const NUM_SYMBOLS: usize = 257;
/// log2 of the total frequency (total = 4096).
pub const FREQ_BITS: u32 = 12;
/// Rescale interval target fixed by the format.
const TARGET_RESCALE: u32 = 2000;

const BOTTOM: u32 = 1 << 23; // renormalization threshold (CODE_BITS - 9 bits)
const EXTRA_BITS: u32 = 7; // (CODE_BITS - 2) % 8 + 1

/// Search-table shift: total is 12 bits, table has 2^7 buckets.
const SEARCH_SHIFT: u32 = FREQ_BITS - 7;

/// Byte-oriented range decoder over an in-memory payload.
#[derive(Debug)]
pub struct RangeDecoder {
    low: u32,
    range: u32,
    help: u32,
    buffer: u8,
    pos: usize,
}

impl RangeDecoder {
    /// Start decoding: consumes the header byte the encoder wrote first
    /// (its value is arbitrary) plus the first payload byte.
    pub fn new(data: &[u8], pos: &mut usize) -> Self {
        let mut dec = Self {
            low: 0,
            range: 1 << EXTRA_BITS,
            help: 0,
            buffer: 0,
            pos: *pos,
        };
        let _header = dec.in_byte(data);
        dec.buffer = dec.in_byte(data);
        dec.low = u32::from(dec.buffer) >> (8 - EXTRA_BITS);
        *pos = dec.pos;
        dec
    }

    #[inline]
    fn in_byte(&mut self, data: &[u8]) -> u8 {
        let b = data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    #[inline]
    fn normalize(&mut self, data: &[u8]) {
        while self.range <= BOTTOM {
            self.low = (self.low << 8) | ((u32::from(self.buffer) << EXTRA_BITS) & 0xff);
            self.buffer = self.in_byte(data);
            self.low |= u32::from(self.buffer) >> (8 - EXTRA_BITS);
            self.range <<= 8;
        }
    }

    /// Cumulative frequency of the next symbol for a total of `1 << shift`.
    #[inline]
    pub fn decode_culshift(&mut self, data: &[u8], shift: u32) -> u32 {
        self.normalize(data);
        self.help = self.range >> shift;
        let tmp = self.low / self.help;
        if tmp >> shift != 0 {
            (1 << shift) - 1
        } else {
            tmp
        }
    }

    /// Narrow the interval to the decoded symbol's frequency band.
    #[inline]
    pub fn decode_update(&mut self, sy_f: u32, lt_f: u32, tot_f: u32) {
        let tmp = self.help * lt_f;
        self.low -= tmp;
        if lt_f + sy_f < tot_f {
            self.range = self.help * sy_f;
        } else {
            self.range -= tmp;
        }
    }
}

/// Quasistatic frequency model: adapts by accumulating per-symbol
/// increments and periodically folding them into the cumulative table.
#[derive(Debug)]
pub struct QsModel {
    /// Cumulative frequencies, `cf[NUM_SYMBOLS] == 4096`.
    cf: Vec<u32>,
    /// Pending (next-interval) frequencies.
    newf: Vec<u32>,
    /// Bucketized reverse index for symbol lookup.
    search: Vec<u16>,
    rescale: u32,
    next_left: u32,
    left: i64,
    incr: u32,
}

impl QsModel {
    /// Fresh model with the uniform initial distribution the format uses.
    pub fn new() -> Self {
        let n = NUM_SYMBOLS;
        let total = 1u32 << FREQ_BITS;
        let mut m = Self {
            cf: vec![0; n + 1],
            newf: vec![0; n + 1],
            search: vec![0; (1 << 7) + 1],
            rescale: (n as u32 >> 4) | 2,
            next_left: 0,
            left: 0,
            incr: 0,
        };
        m.cf[n] = total;
        m.search[1 << 7] = (n - 1) as u16;
        let initval = total / n as u32;
        let extra = (total % n as u32) as usize;
        for i in 0..n {
            m.newf[i] = if i < extra { initval + 1 } else { initval };
        }
        m.rescale_now();
        m
    }

    fn rescale_now(&mut self) {
        if self.next_left != 0 {
            // Half-step: bump the increment and run out the remainder
            // before doing a real rescale.
            self.incr += 1;
            self.left = i64::from(self.next_left);
            self.next_left = 0;
            return;
        }
        if self.rescale < TARGET_RESCALE {
            self.rescale = (self.rescale << 1).min(TARGET_RESCALE);
        }
        let n = NUM_SYMBOLS;
        let total = self.cf[n];
        let mut cf = total;
        let mut missing = total;
        for i in (1..n).rev() {
            let tmp = self.newf[i];
            cf -= tmp;
            self.cf[i] = cf;
            let halved = (tmp >> 1) | 1;
            missing -= halved;
            self.newf[i] = halved;
        }
        debug_assert_eq!(cf, self.newf[0]);
        self.newf[0] = (self.newf[0] >> 1) | 1;
        missing -= self.newf[0];
        self.incr = missing / self.rescale;
        self.next_left = missing % self.rescale;
        self.left = i64::from(self.rescale - self.next_left);

        // Rebuild the bucket index over the fresh cumulative table.
        let mut i = n;
        while i > 0 {
            let end = ((self.cf[i] - 1) >> SEARCH_SHIFT) as usize;
            i -= 1;
            let start = (self.cf[i] >> SEARCH_SHIFT) as usize;
            for slot in start..=end {
                self.search[slot] = i as u16;
            }
        }
    }

    /// Symbol whose frequency band contains cumulative frequency `lt_f`.
    #[inline]
    pub fn get_sym(&self, lt_f: u32) -> usize {
        let bucket = (lt_f >> SEARCH_SHIFT) as usize;
        let mut lo = self.search[bucket] as usize;
        let mut hi = self.search[bucket + 1] as usize + 1;
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if lt_f < self.cf[mid] {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    }

    /// Frequency band of `sym`: returns `(sy_f, lt_f)`.
    #[inline]
    pub fn get_freq(&self, sym: usize) -> (u32, u32) {
        let lt = self.cf[sym];
        (self.cf[sym + 1] - lt, lt)
    }

    /// Record one occurrence of `sym`.
    #[inline]
    pub fn update(&mut self, sym: usize) {
        if self.left <= 0 {
            self.rescale_now();
        }
        self.left -= 1;
        self.newf[sym] += self.incr;
    }
}

/// Decode one 8-bit model symbol (0..=255) or the end marker (256).
#[inline]
pub fn decode_symbol(rc: &mut RangeDecoder, model: &mut QsModel, data: &[u8]) -> usize {
    let ltf = rc.decode_culshift(data, FREQ_BITS);
    let ch = model.get_sym(ltf);
    let (sy, lt) = model.get_freq(ch);
    rc.decode_update(sy, lt, 1 << FREQ_BITS);
    model.update(ch);
    ch
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_initial_distribution_sums_to_total() {
        let m = QsModel::new();
        assert_eq!(m.cf[NUM_SYMBOLS], 1 << FREQ_BITS);
        assert_eq!(m.cf[0], 0);
        for i in 0..NUM_SYMBOLS {
            assert!(m.cf[i] < m.cf[i + 1], "cf must be strictly increasing");
        }
    }

    #[test]
    fn get_sym_inverts_get_freq() {
        let m = QsModel::new();
        for sym in [0usize, 1, 100, 255, 256] {
            let (sy, lt) = m.get_freq(sym);
            assert!(sy > 0);
            assert_eq!(m.get_sym(lt), sym);
            assert_eq!(m.get_sym(lt + sy - 1), sym);
        }
    }
}
