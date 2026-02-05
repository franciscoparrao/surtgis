//! Generic n-band spectral index builder
//!
//! Allows computing arbitrary spectral indices from user-defined formulas.
//! Supports any number of input bands with named references.
//!
//! Example formulas:
//! - `"(NIR - Red) / (NIR + Red)"` → NDVI
//! - `"(NIR - Red) / (NIR + Red + Blue)"` → 3-band index
//! - `"2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)"` → EVI
//!
//! This is a UNIQUE feature — no competitor (WBT, SAGA, GRASS) offers
//! a generic formula-based n-band index builder as a core function.
//!
//! Reference:
//! Wang, F. et al. (2019). Three-band spectral indices outperform
//! two-band for crop phenology. *Field Crops Research*.

use std::collections::HashMap;
use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// A token in the parsed expression
#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Band(String),
    Op(char),    // +, -, *, /
    LParen,
    RParen,
}

/// A node in the expression AST
#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Band(String),
    BinOp {
        op: char,
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
            ' ' | '\t' | '\n' => { i += 1; }
            '+' | '-' | '*' | '/' => {
                tokens.push(Token::Op(chars[i]));
                i += 1;
            }
            '(' => { tokens.push(Token::LParen); i += 1; }
            ')' => { tokens.push(Token::RParen); i += 1; }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num = num_str.parse::<f64>().map_err(|_| {
                    Error::Algorithm(format!("Invalid number: {}", num_str))
                })?;
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
                    "Unexpected character '{}' in formula", c
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
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
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

    /// Parse: term = factor (('*' | '/') factor)*
    fn parse_term(&mut self) -> Result<Expr> {
        let mut left = self.parse_factor()?;

        while let Some(Token::Op(op @ ('*' | '/'))) = self.peek() {
            let op = *op;
            self.advance();
            let right = self.parse_factor()?;
            left = Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse: factor = number | band | '(' expr ')' | '-' factor
    fn parse_factor(&mut self) -> Result<Expr> {
        match self.peek().cloned() {
            Some(Token::Number(n)) => {
                self.advance();
                Ok(Expr::Num(n))
            }
            Some(Token::Band(name)) => {
                self.advance();
                Ok(Expr::Band(name))
            }
            Some(Token::LParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                match self.advance() {
                    Some(Token::RParen) => Ok(expr),
                    _ => Err(Error::Algorithm("Expected closing parenthesis".into())),
                }
            }
            Some(Token::Op('-')) => {
                self.advance();
                let factor = self.parse_factor()?;
                Ok(Expr::Neg(Box::new(factor)))
            }
            Some(Token::Op('+')) => {
                self.advance();
                self.parse_factor()
            }
            other => Err(Error::Algorithm(format!(
                "Unexpected token in formula: {:?}", other
            ))),
        }
    }
}

/// Evaluate an expression with given band values
fn eval(expr: &Expr, bands: &HashMap<String, f64>) -> f64 {
    match expr {
        Expr::Num(n) => *n,
        Expr::Band(name) => {
            *bands.get(name).unwrap_or(&f64::NAN)
        }
        Expr::BinOp { op, left, right } => {
            let l = eval(left, bands);
            let r = eval(right, bands);
            match op {
                '+' => l + r,
                '-' => l - r,
                '*' => l * r,
                '/' => {
                    if r.abs() < 1e-10 { f64::NAN } else { l / r }
                }
                _ => f64::NAN,
            }
        }
        Expr::Neg(inner) => -eval(inner, bands),
    }
}

/// Collect all band names referenced in an expression
fn collect_bands(expr: &Expr, names: &mut Vec<String>) {
    match expr {
        Expr::Band(name) => {
            if !names.contains(name) {
                names.push(name.clone());
            }
        }
        Expr::BinOp { left, right, .. } => {
            collect_bands(left, names);
            collect_bands(right, names);
        }
        Expr::Neg(inner) => collect_bands(inner, names),
        Expr::Num(_) => {}
    }
}

/// Compute a custom spectral index from a formula and named bands.
///
/// # Arguments
/// * `formula` - Arithmetic expression referencing band names.
///   Supports: `+`, `-`, `*`, `/`, parentheses, numeric constants.
///   Example: `"(NIR - Red) / (NIR + Red + Blue)"`
/// * `bands` - Map of band name → raster. All rasters must have
///   the same dimensions.
///
/// # Returns
/// Raster<f64> with the computed index values.
///
/// # Errors
/// - If formula is invalid (parse error)
/// - If a referenced band is not in the map
/// - If raster dimensions don't match
pub fn index_builder(
    formula: &str,
    bands: &HashMap<&str, &Raster<f64>>,
) -> Result<Raster<f64>> {
    if bands.is_empty() {
        return Err(Error::Algorithm("No bands provided".into()));
    }

    // Parse formula
    let tokens = tokenize(formula)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;

    // Validate all referenced bands exist
    let mut referenced = Vec::new();
    collect_bands(&expr, &mut referenced);

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
                er: rows, ec: cols,
                ar: r, ac: c,
            });
        }
    }

    // Evaluate for each pixel
    // Build band data references for parallel access
    let band_names: Vec<String> = bands.keys().map(|s| s.to_string()).collect();
    let band_refs: Vec<&Raster<f64>> = band_names.iter()
        .map(|name| *bands.get(name.as_str()).unwrap())
        .collect();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let mut band_values = HashMap::new();
                let mut any_nan = false;

                for (i, name) in band_names.iter().enumerate() {
                    let val = unsafe { band_refs[i].get_unchecked(row, col) };
                    if val.is_nan() {
                        any_nan = true;
                        break;
                    }
                    band_values.insert(name.clone(), val);
                }

                if any_nan {
                    continue;
                }

                let result = eval(&expr, &band_values);
                *row_data_col = result;
            }

            row_data
        })
        .collect();

    let mut output = first.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_band(rows: usize, cols: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, value);
        r.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
        r
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
        assert!(
            (v - 0.6).abs() < 0.001,
            "NDVI should be 0.6, got {}",
            v
        );
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
            expected, v
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
        assert!(
            (v - 22.5).abs() < 0.001,
            "5.0 * 2.5 + 10 = 22.5, got {}",
            v
        );
    }
}
