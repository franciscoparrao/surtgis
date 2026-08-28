//! Generic n-band spectral index builder
//!
//! Allows computing arbitrary spectral indices from user-defined formulas.
//! Supports any number of input bands with named references.
//!
//! Example formulas:
//! - `"(NIR - Red) / (NIR + Red)"` → NDVI
//! - `"(NIR - Red) / (NIR + Red + Blue)"` → 3-band index
//! - `"2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)"` → EVI
//! - `"1.0 / ((0.1 - Red) ** 2.0 + (0.06 - NIR) ** 2.0)"` → BAI
//!
//! The grammar (`+ - * / **`, parentheses, unary minus, numeric constants)
//! covers every parameter-free formula in the Awesome Spectral Indices
//! catalogue, so the standard band names of that catalogue (`N`, `R`, `G`,
//! `B`, `RE1`..`RE3`, `N2`, `S1`, `S2`, `A`, `WV`) evaluate as-is.
//!
//! References:
//! - Wang, F. et al. (2019). Three-band spectral indices outperform
//!   two-band for crop phenology. *Field Crops Research*.
//! - Montero, D. et al. (2023). A standardized catalogue of spectral
//!   indices to advance the use of remote sensing in Earth system
//!   research. *Scientific Data*, 10, 197.

use crate::maybe_rayon::par_map_rows;
use std::collections::HashMap;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// A token in the parsed expression
#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Band(String),
    Op(char), // +, -, *, /
    Pow,      // **
    LParen,
    RParen,
}

/// A node in the expression AST
///
/// Band references are resolved to indices into the parser's band-name
/// table at parse time, so per-pixel evaluation is a slice lookup.
#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Band(usize),
    BinOp {
        op: char, // +, -, *, /, ^ (power)
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Neg(Box<Expr>),
}

/// Tokenize a formula string
fn tokenize(formula: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = formula.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' => {
                i += 1;
            }
            '*' if i + 1 < chars.len() && chars[i + 1] == '*' => {
                tokens.push(Token::Pow);
                i += 2;
            }
            '+' | '-' | '*' | '/' => {
                tokens.push(Token::Op(chars[i]));
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num = num_str
                    .parse::<f64>()
                    .map_err(|_| Error::Algorithm(format!("Invalid number: {}", num_str)))?;
                tokens.push(Token::Number(num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                tokens.push(Token::Band(name));
            }
            c => {
                return Err(Error::Algorithm(format!(
                    "Unexpected character '{}' in formula",
                    c
                )));
            }
        }
    }

    Ok(tokens)
}

/// Recursive descent parser for arithmetic expressions
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Band names in order of first appearance; `Expr::Band` indexes here.
    band_names: Vec<String>,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            band_names: Vec::new(),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn band_index(&mut self, name: String) -> usize {
        if let Some(idx) = self.band_names.iter().position(|n| *n == name) {
            idx
        } else {
            self.band_names.push(name);
            self.band_names.len() - 1
        }
    }

    /// Parse: expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_term()?;

        while let Some(Token::Op(op @ ('+' | '-'))) = self.peek() {
            let op = *op;
            self.advance();
            let right = self.parse_term()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse: term = unary (('*' | '/') unary)*
    fn parse_term(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary()?;

        while let Some(Token::Op(op @ ('*' | '/'))) = self.peek() {
            let op = *op;
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse: unary = ('-' | '+') unary | power
    fn parse_unary(&mut self) -> Result<Expr> {
        match self.peek() {
            Some(Token::Op('-')) => {
                self.advance();
                let inner = self.parse_unary()?;
                Ok(Expr::Neg(Box::new(inner)))
            }
            Some(Token::Op('+')) => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_power(),
        }
    }

    /// Parse: power = atom ('**' unary)?
    ///
    /// Right-associative, and binds tighter than unary minus on its left:
    /// `-x ** 2` is `-(x ** 2)`, while `x ** -2` is legal (Python rules).
    fn parse_power(&mut self) -> Result<Expr> {
        let base = self.parse_atom()?;

        if let Some(Token::Pow) = self.peek() {
            self.advance();
            let exponent = self.parse_unary()?;
            Ok(Expr::BinOp {
                op: '^',
                left: Box::new(base),
                right: Box::new(exponent),
            })
        } else {
            Ok(base)
        }
    }

    /// Parse: atom = number | band | '(' expr ')'
    fn parse_atom(&mut self) -> Result<Expr> {
        match self.peek().cloned() {
            Some(Token::Number(n)) => {
                self.advance();
                Ok(Expr::Num(n))
            }
            Some(Token::Band(name)) => {
                self.advance();
                let idx = self.band_index(name);
                Ok(Expr::Band(idx))
            }
            Some(Token::LParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                match self.advance() {
                    Some(Token::RParen) => Ok(expr),
                    _ => Err(Error::Algorithm("Expected closing parenthesis".into())),
                }
            }
            other => Err(Error::Algorithm(format!(
                "Unexpected token in formula: {:?}",
                other
            ))),
        }
    }
}

/// Evaluate an expression against band values indexed by `Expr::Band`.
fn eval(expr: &Expr, values: &[f64]) -> f64 {
    match expr {
        Expr::Num(n) => *n,
        Expr::Band(idx) => values[*idx],
        Expr::BinOp { op, left, right } => {
            let l = eval(left, values);
            let r = eval(right, values);
            match op {
                '+' => l + r,
                '-' => l - r,
                '*' => l * r,
                '/' => {
                    if r.abs() < 1e-10 {
                        f64::NAN
                    } else {
                        l / r
                    }
                }
                '^' => l.powf(r),
                _ => f64::NAN,
            }
        }
        Expr::Neg(inner) => -eval(inner, values),
    }
}

/// Compute a custom spectral index from a formula and named bands.
///
/// # Arguments
/// * `formula` - Arithmetic expression referencing band names.
///   Supports: `+`, `-`, `*`, `/`, `**` (right-associative power),
///   parentheses, unary minus and numeric constants — the full grammar of
///   the Awesome Spectral Indices catalogue (Montero et al., 2023).
///   Example: `"(NIR - Red) / (NIR + Red + Blue)"`
/// * `bands` - Map of band name → raster. All rasters must have
///   the same dimensions.
///
/// # Returns
/// `Raster<f64>` with the computed index values. A pixel that is NaN or
/// the declared nodata value in any referenced band is NaN in the output,
/// as is any division by (near-)zero.
///
/// # Errors
/// - If formula is invalid (parse error or trailing tokens)
/// - If a referenced band is not in the map
/// - If raster dimensions don't match
pub fn index_builder(formula: &str, bands: &HashMap<&str, &Raster<f64>>) -> Result<Raster<f64>> {
    if bands.is_empty() {
        return Err(Error::Algorithm("No bands provided".into()));
    }

    // Parse formula
    let tokens = tokenize(formula)?;
    let n_tokens = tokens.len();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    if parser.pos != n_tokens {
        return Err(Error::Algorithm(format!(
            "Unexpected token after end of expression in formula '{}'",
            formula
        )));
    }
    let referenced = parser.band_names;

    // Validate all referenced bands exist
    for name in &referenced {
        if !bands.contains_key(name.as_str()) {
            return Err(Error::Algorithm(format!(
                "Band '{}' not found. Available: {:?}",
                name,
                bands.keys().collect::<Vec<_>>()
            )));
        }
    }

    // Get dimensions from first band
    let first = *bands.values().next().unwrap();
    let (rows, cols) = first.shape();

    // Verify all bands have same dimensions
    for raster in bands.values() {
        let (r, c) = raster.shape();
        if r != rows || c != cols {
            return Err(Error::SizeMismatch {
                er: rows,
                ec: cols,
                ar: r,
                ac: c,
            });
        }
    }

    // Band rasters in `Expr::Band` index order; metadata from the first
    // referenced band so the output georeferencing is deterministic.
    let band_refs: Vec<&Raster<f64>> = referenced
        .iter()
        .map(|name| *bands.get(name.as_str()).unwrap())
        .collect();
    let nodatas: Vec<Option<f64>> = band_refs.iter().map(|r| r.nodata()).collect();
    let n_bands = band_refs.len();
    let meta_src = band_refs.first().copied().unwrap_or(first);

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        let mut values = vec![0.0_f64; n_bands];

        'cell: for (col, out_val) in out_row.iter_mut().enumerate() {
            for (i, band) in band_refs.iter().enumerate() {
                let val = unsafe { band.get_unchecked(row, col) };
                if val.is_nan() {
                    continue 'cell;
                }
                if let Some(nd) = nodatas[i]
                    && val == nd
                {
                    continue 'cell;
                }
                values[i] = val;
            }

            *out_val = eval(&expr, &values);
        }
    });

    let mut output = meta_src.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imagery::{SaviParams, gndvi, mndwi, nbr, ndmi, ndvi, savi};
    use surtgis_core::GeoTransform;

    fn make_band(rows: usize, cols: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, value);
        r.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
        r
    }

    /// Reflectance-like values varying per cell, including negatives.
    fn make_varied(rows: usize, cols: usize, seed: f64) -> Raster<f64> {
        let mut r = make_band(rows, cols, 0.0);
        for i in 0..rows {
            for j in 0..cols {
                let v = ((i * cols + j) as f64 * 0.37 + seed).sin() * 0.5;
                r.set(i, j, v).unwrap();
            }
        }
        r
    }

    fn assert_bit_identical(a: &Raster<f64>, b: &Raster<f64>) {
        let (rows, cols) = a.shape();
        assert_eq!((rows, cols), b.shape());
        for i in 0..rows {
            for j in 0..cols {
                let va = a.get(i, j).unwrap();
                let vb = b.get(i, j).unwrap();
                assert!(
                    va.to_bits() == vb.to_bits(),
                    "bit mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    va,
                    vb
                );
            }
        }
    }

    /// Band pair with a NaN pixel and an exact zero-sum pixel, to exercise
    /// nodata propagation and the division guard in parity tests.
    fn parity_bands() -> (Raster<f64>, Raster<f64>) {
        let mut a = make_varied(8, 9, 0.1);
        let mut b = make_varied(8, 9, 2.3);
        a.set(1, 1, f64::NAN).unwrap();
        a.set(2, 2, 0.3).unwrap();
        b.set(2, 2, -0.3).unwrap();
        (a, b)
    }

    #[test]
    fn test_ndvi_formula() {
        let nir = make_band(5, 5, 0.8);
        let red = make_band(5, 5, 0.2);

        let mut bands = HashMap::new();
        bands.insert("NIR", &nir);
        bands.insert("Red", &red);

        let result = index_builder("(NIR - Red) / (NIR + Red)", &bands).unwrap();
        let v = result.get(2, 2).unwrap();

        // NDVI = (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        assert!((v - 0.6).abs() < 0.001, "NDVI should be 0.6, got {}", v);
    }

    #[test]
    fn test_three_band_index() {
        let nir = make_band(5, 5, 0.9);
        let red = make_band(5, 5, 0.3);
        let blue = make_band(5, 5, 0.1);

        let mut bands = HashMap::new();
        bands.insert("NIR", &nir);
        bands.insert("Red", &red);
        bands.insert("Blue", &blue);

        let result = index_builder("(NIR - Red) / (NIR + Red + Blue)", &bands).unwrap();
        let v = result.get(2, 2).unwrap();

        // (0.9 - 0.3) / (0.9 + 0.3 + 0.1) = 0.6 / 1.3 ≈ 0.4615
        assert!(
            (v - 0.4615).abs() < 0.01,
            "3-band index should be ~0.46, got {}",
            v
        );
    }

    #[test]
    fn test_evi_formula() {
        let nir = make_band(3, 3, 0.8);
        let red = make_band(3, 3, 0.2);
        let blue = make_band(3, 3, 0.1);

        let mut bands = HashMap::new();
        bands.insert("NIR", &nir);
        bands.insert("Red", &red);
        bands.insert("Blue", &blue);

        let formula = "2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)";
        let result = index_builder(formula, &bands).unwrap();
        let v = result.get(1, 1).unwrap();

        // 2.5 * (0.8-0.2) / (0.8 + 1.2 - 0.75 + 1) = 2.5*0.6/2.25 ≈ 0.6667
        let expected = 2.5 * 0.6 / (0.8 + 1.2 - 0.75 + 1.0);
        assert!(
            (v - expected).abs() < 0.01,
            "EVI should be ~{:.4}, got {}",
            expected,
            v
        );
    }

    #[test]
    fn test_missing_band_error() {
        let nir = make_band(3, 3, 0.8);
        let mut bands = HashMap::new();
        bands.insert("NIR", &nir);

        let result = index_builder("(NIR - Red) / (NIR + Red)", &bands);
        assert!(result.is_err(), "Should error on missing band");
    }

    #[test]
    fn test_invalid_formula_error() {
        let nir = make_band(3, 3, 0.8);
        let mut bands = HashMap::new();
        bands.insert("NIR", &nir);

        let result = index_builder("(NIR - ", &bands);
        assert!(result.is_err(), "Should error on invalid formula");
    }

    #[test]
    fn test_trailing_tokens_error() {
        let nir = make_band(3, 3, 0.8);
        let mut bands = HashMap::new();
        bands.insert("NIR", &nir);

        let result = index_builder("NIR NIR", &bands);
        assert!(result.is_err(), "Should error on trailing tokens");
    }

    #[test]
    fn test_division_by_zero_returns_nan() {
        let a = make_band(3, 3, 1.0);
        let b = make_band(3, 3, 0.0);

        let mut bands = HashMap::new();
        bands.insert("A", &a);
        bands.insert("B", &b);

        let result = index_builder("A / B", &bands).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!(v.is_nan(), "Division by zero should produce NaN");
    }

    #[test]
    fn test_constant_expression() {
        let a = make_band(3, 3, 5.0);
        let mut bands = HashMap::new();
        bands.insert("A", &a);

        let result = index_builder("A * 2.5 + 10", &bands).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!((v - 22.5).abs() < 0.001, "5.0 * 2.5 + 10 = 22.5, got {}", v);
    }

    #[test]
    fn test_declared_nodata_propagates() {
        let mut a = make_band(3, 3, 0.8);
        a.set_nodata(Some(-9999.0));
        a.set(1, 1, -9999.0).unwrap();
        let b = make_band(3, 3, 0.2);

        let mut bands = HashMap::new();
        bands.insert("A", &a);
        bands.insert("B", &b);

        let result = index_builder("(A - B) / (A + B)", &bands).unwrap();
        assert!(result.get(1, 1).unwrap().is_nan());
        assert!((result.get(0, 0).unwrap() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_power_right_associative() {
        let a = make_band(3, 3, 2.0);
        let mut bands = HashMap::new();
        bands.insert("A", &a);

        // Right-associative: 2 ** (3 ** 2) = 2^9 = 512, not (2**3)**2 = 64
        let result = index_builder("A ** 3.0 ** 2.0", &bands).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!((v - 512.0).abs() < 1e-9, "2**3**2 should be 512, got {}", v);
    }

    #[test]
    fn test_power_binds_tighter_than_unary_minus() {
        let a = make_band(3, 3, 3.0);
        let mut bands = HashMap::new();
        bands.insert("A", &a);

        let result = index_builder("-A ** 2.0", &bands).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!((v + 9.0).abs() < 1e-12, "-3**2 should be -9, got {}", v);

        let result = index_builder("(0.0 - A) ** 2.0", &bands).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!((v - 9.0).abs() < 1e-12, "(-3)**2 should be 9, got {}", v);
    }

    #[test]
    fn test_power_negative_exponent() {
        let a = make_band(3, 3, 2.0);
        let mut bands = HashMap::new();
        bands.insert("A", &a);

        let result = index_builder("A ** -2.0", &bands).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!((v - 0.25).abs() < 1e-12, "2**-2 should be 0.25, got {}", v);
    }

    #[test]
    fn test_bai_formula() {
        // BAI from the Awesome Spectral Indices catalogue.
        let red = make_band(3, 3, 0.08);
        let nir = make_band(3, 3, 0.2);

        let mut bands = HashMap::new();
        bands.insert("R", &red);
        bands.insert("N", &nir);

        let formula = "1.0 / ((0.1 - R) ** 2.0 + (0.06 - N) ** 2.0)";
        let result = index_builder(formula, &bands).unwrap();
        let v = result.get(1, 1).unwrap();

        // 1 / (0.02² + (-0.14)²) = 1 / 0.02 = 50
        let expected = 1.0 / (0.02_f64.powf(2.0) + (-0.14_f64).powf(2.0));
        assert!(
            (v - expected).abs() < 1e-9,
            "BAI should be {}, got {}",
            expected,
            v
        );
    }

    // -- Parity with the hand-written indices (bit-identical) ---------------

    #[test]
    fn test_parity_ndvi() {
        let (nir, red) = parity_bands();
        let by_hand = ndvi(&nir, &red).unwrap();

        let mut bands = HashMap::new();
        bands.insert("N", &nir);
        bands.insert("R", &red);
        let by_formula = index_builder("(N - R) / (N + R)", &bands).unwrap();

        assert_bit_identical(&by_formula, &by_hand);
    }

    #[test]
    fn test_parity_nbr() {
        let (nir, swir2) = parity_bands();
        let by_hand = nbr(&nir, &swir2).unwrap();

        let mut bands = HashMap::new();
        bands.insert("N", &nir);
        bands.insert("S2", &swir2);
        let by_formula = index_builder("(N - S2) / (N + S2)", &bands).unwrap();

        assert_bit_identical(&by_formula, &by_hand);
    }

    #[test]
    fn test_parity_mndwi() {
        let (green, swir1) = parity_bands();
        let by_hand = mndwi(&green, &swir1).unwrap();

        let mut bands = HashMap::new();
        bands.insert("G", &green);
        bands.insert("S1", &swir1);
        let by_formula = index_builder("(G - S1) / (G + S1)", &bands).unwrap();

        assert_bit_identical(&by_formula, &by_hand);
    }

    #[test]
    fn test_parity_ndmi() {
        let (nir, swir1) = parity_bands();
        let by_hand = ndmi(&nir, &swir1).unwrap();

        let mut bands = HashMap::new();
        bands.insert("N", &nir);
        bands.insert("S1", &swir1);
        let by_formula = index_builder("(N - S1) / (N + S1)", &bands).unwrap();

        assert_bit_identical(&by_formula, &by_hand);
    }

    #[test]
    fn test_parity_gndvi() {
        let (nir, green) = parity_bands();
        let by_hand = gndvi(&nir, &green).unwrap();

        let mut bands = HashMap::new();
        bands.insert("N", &nir);
        bands.insert("G", &green);
        let by_formula = index_builder("(N - G) / (N + G)", &bands).unwrap();

        assert_bit_identical(&by_formula, &by_hand);
    }

    #[test]
    fn test_parity_savi() {
        let (nir, red) = parity_bands();
        let by_hand = savi(&nir, &red, SaviParams { l_factor: 0.5 }).unwrap();

        let mut bands = HashMap::new();
        bands.insert("N", &nir);
        bands.insert("R", &red);
        // Same operation order as the hand-written savi(): the L parameter
        // substituted into the formula, as the ASI convention prescribes.
        let by_formula = index_builder("((N - R) / (N + R + 0.5)) * (1.0 + 0.5)", &bands).unwrap();

        assert_bit_identical(&by_formula, &by_hand);
    }
}
