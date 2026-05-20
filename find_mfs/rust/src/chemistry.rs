pub fn bond_electrons(symbol: &str) -> Option<i32> {
    match symbol {
        "H" | "Li" | "Na" | "F" | "Cl" | "Br" | "I" => Some(1),
        "C" | "Si" => Some(4),
        "N" | "B" => Some(3),
        "O" | "S" => Some(2),
        "P" => Some(5),
        _ => None,
    }
}

pub fn rdbe_coeff_for_symbol(symbol: &str) -> f64 {
    let electrons = bond_electrons(symbol).unwrap_or(2);
    0.5 * ((electrons - 2) as f64)
}

#[cfg(test)]
pub fn rdbe_from_counts_i64(counts: &[i64], rdbe_coeffs: &[f64]) -> f64 {
    let mut rdbe = 1.0;
    for (count, coeff) in counts.iter().zip(rdbe_coeffs.iter()) {
        rdbe += (*count as f64) * coeff;
    }
    rdbe
}

#[cfg(test)]
mod tests {
    use super::{
        bond_electrons, rdbe_coeff_for_symbol, rdbe_from_counts_i32, rdbe_from_counts_i64,
    };

    #[test]
    fn bond_electron_table_matches_supported_python_elements() {
        assert_eq!(bond_electrons("C"), Some(4));
        assert_eq!(bond_electrons("H"), Some(1));
        assert_eq!(bond_electrons("Cl"), Some(1));
        assert_eq!(bond_electrons("Xe"), None);
        assert_eq!(rdbe_coeff_for_symbol("C"), 1.0);
        assert_eq!(rdbe_coeff_for_symbol("H"), -0.5);
        assert_eq!(rdbe_coeff_for_symbol("Xe"), 0.0);
    }

    #[test]
    fn rdbe_helpers_sum_coefficients() {
        assert_eq!(rdbe_from_counts_i64(&[6, 6], &[1.0, -0.5]), 4.0);
        assert_eq!(rdbe_from_counts_i32(&[6, 6], &[1.0, -0.5]), 4.0);
    }
}

pub fn rdbe_from_counts_i32(counts: &[i32], rdbe_coeffs: &[f64]) -> f64 {
    let mut rdbe = 1.0;
    for (count, coeff) in counts.iter().zip(rdbe_coeffs.iter()) {
        rdbe += (*count as f64) * coeff;
    }
    rdbe
}
