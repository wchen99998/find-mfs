pub fn passes_rdbe_and_octet(
    rdbe: f64,
    rdbe_min: f64,
    rdbe_max: f64,
    check_octet: bool,
    charge_parity_even: bool,
) -> bool {
    if rdbe < rdbe_min || rdbe > rdbe_max {
        return false;
    }

    if !check_octet {
        return true;
    }

    let doubled_int = (2.0 * rdbe) as i64;
    let is_half_int = (doubled_int & 1) == 1;

    if charge_parity_even {
        !is_half_int
    } else {
        is_half_int
    }
}

pub fn passes_residual_octet(rdbe: f64, charge_parity_even: bool) -> bool {
    let doubled_int = (2.0 * rdbe).round() as i64;
    let is_half_int = (doubled_int & 1) == 1;

    if charge_parity_even {
        !is_half_int
    } else {
        is_half_int
    }
}

#[cfg(test)]
mod tests {
    use super::{passes_rdbe_and_octet, passes_residual_octet};

    #[test]
    fn rdbe_range_and_octet_match_expected_parity() {
        assert!(passes_rdbe_and_octet(4.0, 0.0, 10.0, true, true));
        assert!(!passes_rdbe_and_octet(4.5, 0.0, 10.0, true, true));
        assert!(passes_rdbe_and_octet(4.5, 0.0, 10.0, true, false));
        assert!(!passes_rdbe_and_octet(-1.0, 0.0, 10.0, false, true));
    }

    #[test]
    fn residual_octet_uses_rounded_double_rdbe() {
        assert!(passes_residual_octet(3.999999999, true));
        assert!(!passes_residual_octet(4.5, true));
        assert!(passes_residual_octet(4.5, false));
    }
}
