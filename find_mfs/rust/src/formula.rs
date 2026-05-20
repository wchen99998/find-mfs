use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
enum ParsedCount {
    Count(i64),
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAdduct {
    pub mass: f64,
    pub symbols: Vec<String>,
    pub counts: Vec<i32>,
}

fn hill_sort_key(symbol: &str) -> (u8, &str) {
    match symbol {
        "C" => (0, ""),
        "H" => (1, ""),
        _ => (2, symbol),
    }
}

pub fn format_formula_from_counts(symbols: &[String], counts: &[i64], charge: i32) -> String {
    let mut nonzero: Vec<(&str, i64)> = symbols
        .iter()
        .zip(counts.iter())
        .filter_map(|(symbol, count)| {
            if *count > 0 {
                Some((symbol.as_str(), *count))
            } else {
                None
            }
        })
        .collect();

    nonzero.sort_by(|(a, _), (b, _)| hill_sort_key(a).cmp(&hill_sort_key(b)));

    let mut base = String::new();
    for (symbol, count) in nonzero {
        base.push_str(symbol);
        if count != 1 {
            base.push_str(&count.to_string());
        }
    }

    if charge == 0 {
        return base;
    }

    let sign = if charge > 0 { '+' } else { '-' };
    let abs_charge = charge.abs();
    if abs_charge == 1 {
        format!("[{}]{}", base, sign)
    } else {
        format!("[{}]{}{}", base, abs_charge, sign)
    }
}

pub fn parse_formula_counts(formula: &str, allowed_symbols: &[String]) -> Result<Vec<i64>, String> {
    let tokens = parse_formula_tokens(formula, false)?;
    let allowed_index: HashMap<&str, usize> = allowed_symbols
        .iter()
        .enumerate()
        .map(|(idx, sym)| (sym.as_str(), idx))
        .collect();
    let mut counts = vec![0_i64; allowed_symbols.len()];

    for (symbol, parsed_count) in tokens {
        let ParsedCount::Count(count) = parsed_count else {
            return Err(format!(
                "wildcard count is not allowed for element '{}'",
                symbol
            ));
        };
        let idx = allowed_index
            .get(symbol.as_str())
            .ok_or_else(|| format!("element '{}' is not in the given element set", symbol))?;
        counts[*idx] += count;
    }

    Ok(counts)
}

pub fn parse_element_symbols(formula: &str) -> Result<Vec<String>, String> {
    let tokens = parse_formula_tokens(formula, false)?;
    let mut symbols = Vec::new();
    for (symbol, _) in tokens {
        if !symbols.iter().any(|existing| existing == &symbol) {
            symbols.push(symbol);
        }
    }
    Ok(symbols)
}

pub fn parse_formula_bounds(
    formula: &str,
    allowed_symbols: &[String],
    wildcard_value: f64,
) -> Result<Vec<f64>, String> {
    let tokens = parse_formula_tokens(formula, true)?;
    let allowed_index: HashMap<&str, usize> = allowed_symbols
        .iter()
        .enumerate()
        .map(|(idx, sym)| (sym.as_str(), idx))
        .collect();
    let mut counts = vec![0.0_f64; allowed_symbols.len()];

    for (symbol, parsed_count) in tokens {
        let idx = allowed_index
            .get(symbol.as_str())
            .ok_or_else(|| format!("element '{}' is not in the given element set", symbol))?;
        counts[*idx] = match parsed_count {
            ParsedCount::Count(count) => count as f64,
            ParsedCount::Wildcard => wildcard_value,
        };
    }

    Ok(counts)
}

pub fn parse_adduct(
    adduct: &str,
    element_masses: &HashMap<String, f64>,
) -> Result<ParsedAdduct, String> {
    if adduct.contains('+') {
        return Err(
            "Adduct string must not contain '+'. Specify charge separately using the 'charge' parameter."
                .to_string(),
        );
    }

    let (sign, formula) = if let Some(stripped) = adduct.strip_prefix('-') {
        (-1_i32, stripped)
    } else {
        (1_i32, adduct)
    };
    if formula.is_empty() {
        return Err("adduct formula must not be empty".to_string());
    }

    let tokens = parse_formula_tokens(formula, false)?;
    let mut order = Vec::new();
    let mut counts_by_symbol: HashMap<String, i32> = HashMap::new();
    let mut mass = 0.0;

    for (symbol, parsed_count) in tokens {
        let ParsedCount::Count(count_i64) = parsed_count else {
            return Err(format!(
                "wildcard count is not allowed for element '{}'",
                symbol
            ));
        };
        let Some(element_mass) = element_masses.get(symbol.as_str()) else {
            return Err(format!("element '{}' is not in the mass table", symbol));
        };
        let count_i32 = i32::try_from(count_i64)
            .map_err(|_| format!("adduct count for element '{}' is too large", symbol))?;
        if !counts_by_symbol.contains_key(symbol.as_str()) {
            order.push(symbol.clone());
        }
        *counts_by_symbol.entry(symbol).or_insert(0) += sign * count_i32;
        mass += (sign as f64) * *element_mass * (count_i64 as f64);
    }

    let mut symbols = Vec::new();
    let mut counts = Vec::new();
    for symbol in order {
        let count = counts_by_symbol[&symbol];
        if count != 0 {
            symbols.push(symbol);
            counts.push(count);
        }
    }

    Ok(ParsedAdduct {
        mass,
        symbols,
        counts,
    })
}

fn parse_formula_tokens(
    formula: &str,
    allow_wildcard: bool,
) -> Result<Vec<(String, ParsedCount)>, String> {
    if formula.is_empty() {
        return Err("formula must not be empty".to_string());
    }

    let chars: Vec<char> = formula.chars().collect();
    let mut i = 0;
    let mut tokens = Vec::new();

    while i < chars.len() {
        let ch = chars[i];
        if !ch.is_ascii_uppercase() {
            return Err(format!("invalid formula syntax at byte {}", i));
        }

        let mut symbol = String::new();
        symbol.push(ch);
        i += 1;

        if i < chars.len() && chars[i].is_ascii_lowercase() {
            symbol.push(chars[i]);
            i += 1;
        }

        let start_digits = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }

        let count = if start_digits != i {
            ParsedCount::Count(
                formula[start_digits..i]
                    .parse::<i64>()
                    .map_err(|_| format!("invalid count for element '{}'", symbol))?,
            )
        } else if allow_wildcard && i < chars.len() && chars[i] == '*' {
            i += 1;
            ParsedCount::Wildcard
        } else {
            ParsedCount::Count(1)
        };

        tokens.push((symbol, count));
    }

    Ok(tokens)
}

pub fn parse_formula_min_bounds(
    formula: &str,
    allowed_symbols: &[String],
) -> Result<Vec<i64>, String> {
    parse_formula_bounds(formula, allowed_symbols, 0.0)?
        .into_iter()
        .map(|value| {
            if value.is_infinite() {
                Ok(0)
            } else if value < i64::MIN as f64 || value > i64::MAX as f64 {
                Err("formula count is outside the supported integer range".to_string())
            } else {
                Ok(value as i64)
            }
        })
        .collect()
}

pub fn parse_formula_max_bounds(
    formula: &str,
    allowed_symbols: &[String],
) -> Result<Vec<f64>, String> {
    parse_formula_bounds(formula, allowed_symbols, f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{
        format_formula_from_counts, parse_adduct, parse_element_symbols, parse_formula_counts,
        parse_formula_max_bounds, parse_formula_min_bounds,
    };

    #[test]
    fn formats_hill_order_and_charge() {
        let symbols = vec!["O".to_string(), "H".to_string(), "C".to_string()];
        assert_eq!(
            format_formula_from_counts(&symbols, &[6, 12, 6], 0),
            "C6H12O6"
        );
        assert_eq!(
            format_formula_from_counts(&symbols, &[6, 12, 6], 1),
            "[C6H12O6]+"
        );
        assert_eq!(
            format_formula_from_counts(&symbols, &[6, 12, 6], -2),
            "[C6H12O6]2-"
        );
    }

    #[test]
    fn parses_counts_in_allowed_symbol_order() {
        let symbols = vec![
            "C".to_string(),
            "H".to_string(),
            "N".to_string(),
            "O".to_string(),
        ];
        let counts = parse_formula_counts("OH2", &symbols).unwrap();
        assert_eq!(counts, vec![0, 2, 0, 1]);
        let zero = parse_formula_counts(
            "C20H40P0",
            &["C".to_string(), "H".to_string(), "P".to_string()],
        )
        .unwrap();
        assert_eq!(zero, vec![20, 40, 0]);
    }

    #[test]
    fn parses_unique_element_symbols_in_formula_order() {
        assert_eq!(
            parse_element_symbols("C6H12O6").unwrap(),
            vec!["C", "H", "O"]
        );
        assert_eq!(
            parse_element_symbols("CHNOPSClBr").unwrap(),
            vec!["C", "H", "N", "O", "P", "S", "Cl", "Br"]
        );
    }

    #[test]
    fn parses_bounds_with_wildcards() {
        let symbols = vec![
            "C".to_string(),
            "H".to_string(),
            "N".to_string(),
            "O".to_string(),
            "P".to_string(),
            "S".to_string(),
        ];
        assert_eq!(
            parse_formula_min_bounds("C5O*", &symbols).unwrap(),
            vec![5, 0, 0, 0, 0, 0]
        );
        let max = parse_formula_max_bounds("C10H20N*S0P0", &symbols).unwrap();
        assert_eq!(max[0], 10.0);
        assert_eq!(max[1], 20.0);
        assert!(max[2].is_infinite());
        assert_eq!(max[4], 0.0);
        assert_eq!(max[5], 0.0);
    }

    #[test]
    fn parses_signed_adducts_with_mass_offsets() {
        let masses = HashMap::from([
            ("H".to_string(), 1.00782503223),
            ("N".to_string(), 14.00307400443),
            ("Na".to_string(), 22.9897692820),
        ]);

        let sodium = parse_adduct("Na", &masses).unwrap();
        assert!((sodium.mass - 22.9897692820).abs() < 1e-12);
        assert_eq!(sodium.symbols, vec!["Na"]);
        assert_eq!(sodium.counts, vec![1]);

        let deprotonated = parse_adduct("-H", &masses).unwrap();
        assert!((deprotonated.mass + 1.00782503223).abs() < 1e-12);
        assert_eq!(deprotonated.symbols, vec!["H"]);
        assert_eq!(deprotonated.counts, vec![-1]);

        let ammonium = parse_adduct("NH4", &masses).unwrap();
        assert!((ammonium.mass - (14.00307400443 + 4.0 * 1.00782503223)).abs() < 1e-12);
        assert_eq!(ammonium.symbols, vec!["N", "H"]);
        assert_eq!(ammonium.counts, vec![1, 4]);
    }
}
