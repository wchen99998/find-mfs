use std::cmp::Ordering;
use std::sync::Arc;

use crate::chemistry;
use crate::filters;
use crate::isotope;

pub struct BuiltDecomposer {
    pub element_symbols: Vec<String>,
    pub real_masses: Vec<f64>,
    pub integer_masses: Vec<i64>,
    pub ert: Vec<Vec<f64>>,
    pub precision: f64,
    pub min_error: f64,
    pub max_error: f64,
}

pub struct DecomposeInput {
    pub ert: Arc<Vec<Vec<f64>>>,
    pub integer_masses: Arc<Vec<i64>>,
    pub real_masses: Arc<Vec<f64>>,
    pub bounds: Vec<f64>,
    pub min_values: Vec<i64>,
    pub min_int: i64,
    pub max_int: i64,
    pub original_min_mass: f64,
    pub original_max_mass: f64,
    pub charge_mass_offset: f64,
    pub max_results: i32,
    pub rdbe_coeffs: Vec<f64>,
    pub rdbe_min: f64,
    pub rdbe_max: f64,
    pub check_octet: bool,
    pub charge_parity_even: bool,
    pub do_rdbe_filter: bool,
    pub do_iso_filter: bool,
    pub iso_m1_coeffs: Vec<f64>,
    pub iso_m2_direct: Vec<f64>,
    pub obs_m1_ratio: f64,
    pub obs_m2_ratio: f64,
    pub iso_tol_rel: f64,
    pub iso_tol_abs: f64,
    pub query_mass: f64,
    pub adduct_mass: f64,
    pub compute_rdbe: bool,
}

pub fn build_decomposer_from_masses(
    mut element_symbols: Vec<String>,
    mut real_masses: Vec<f64>,
) -> Result<BuiltDecomposer, String> {
    if element_symbols.is_empty() {
        return Err("at least one element is required".to_string());
    }
    if element_symbols.len() != real_masses.len() {
        return Err("element symbols and masses must have the same length".to_string());
    }

    let mut paired: Vec<(String, f64)> = element_symbols
        .drain(..)
        .zip(real_masses.drain(..))
        .collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
    let element_symbols: Vec<String> = paired.iter().map(|(symbol, _)| symbol.clone()).collect();
    let real_masses: Vec<f64> = paired.iter().map(|(_, mass)| *mass).collect();

    let mut precision = 1.0 / 5963.337687_f64;
    let mut integer_masses: Vec<i64> = real_masses
        .iter()
        .map(|mass| (*mass / precision) as i64)
        .collect();

    if integer_masses.len() == 1 {
        let d = integer_masses[0];
        precision *= d as f64;
        integer_masses[0] = 1;
    } else if integer_masses.len() > 1 {
        let mut d = gcd(integer_masses[0], integer_masses[1]);
        for mass in integer_masses.iter().skip(2) {
            d = gcd(d, *mass);
            if d == 1 {
                break;
            }
        }
        if d > 1 {
            precision *= d as f64;
            for mass in &mut integer_masses {
                *mass /= d;
            }
        }
    }

    let ert = calculate_ert(&integer_masses)?;
    let (min_error, max_error) = compute_errors(&real_masses, &integer_masses, precision);

    Ok(BuiltDecomposer {
        element_symbols,
        real_masses,
        integer_masses,
        ert,
        precision,
        min_error,
        max_error,
    })
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn calculate_ert(integer_masses: &[i64]) -> Result<Vec<Vec<f64>>, String> {
    let first_mass = integer_masses[0];
    if first_mass <= 0 {
        return Err("first integer mass must be positive".to_string());
    }
    let first_mass_usize = first_mass as usize;
    let n_elements = integer_masses.len();
    let mut ert = vec![vec![f64::INFINITY; n_elements]; first_mass_usize];
    ert[0][0] = 0.0;

    for j in 1..n_elements {
        ert[0][j] = 0.0;
        let current_mass = integer_masses[j];
        let d = gcd(first_mass, current_mass);

        for p in 0..d {
            let mut n;
            if p == 0 {
                n = 0_i64;
            } else {
                let mut best = f64::INFINITY;
                let mut argmin = p as usize;
                let mut i = p as usize;
                while i < first_mass_usize {
                    if ert[i][j - 1] < best {
                        best = ert[i][j - 1];
                        argmin = i;
                    }
                    i += d as usize;
                }

                if best.is_infinite() {
                    let mut i = p as usize;
                    while i < first_mass_usize {
                        ert[i][j] = f64::INFINITY;
                        i += d as usize;
                    }
                    continue;
                }

                ert[argmin][j] = best;
                n = best as i64;
            }

            for _ in 1..(first_mass / d) {
                n += current_mass;
                let r = (n % first_mass) as usize;
                n = (n as f64).min(ert[r][j - 1]) as i64;
                ert[r][j] = n as f64;
            }
        }
    }

    Ok(ert)
}

fn compute_errors(real_masses: &[f64], integer_masses: &[i64], precision: f64) -> (f64, f64) {
    let mut min_error = 0.0_f64;
    let mut max_error = 0.0_f64;

    for (real_mass, integer_mass) in real_masses.iter().zip(integer_masses.iter()) {
        let error = (precision * (*integer_mass as f64) - *real_mass) / *real_mass;
        min_error = min_error.min(error);
        max_error = max_error.max(error);
    }

    (min_error, max_error)
}

pub struct DecomposeOutput {
    pub counts: Vec<Vec<i32>>,
    pub exact_masses: Vec<f64>,
    pub error_ppm: Vec<f64>,
    pub error_da: Vec<f64>,
    pub rdbe: Option<Vec<f64>>,
}

impl DecomposeOutput {
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn retain_by_mask(&mut self, mask: &[bool]) {
        self.counts = retain_vec_by_mask(&self.counts, mask);
        self.exact_masses = retain_vec_by_mask(&self.exact_masses, mask);
        self.error_ppm = retain_vec_by_mask(&self.error_ppm, mask);
        self.error_da = retain_vec_by_mask(&self.error_da, mask);
        if let Some(rdbe) = self.rdbe.as_mut() {
            *rdbe = retain_vec_by_mask(rdbe, mask);
        }
    }
}

pub fn retain_vec_by_mask<T: Clone>(values: &[T], mask: &[bool]) -> Vec<T> {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, keep)| if *keep { Some(value.clone()) } else { None })
        .collect()
}

pub fn validate_input(input: &DecomposeInput) -> Result<(), String> {
    let num_elements = input.integer_masses.len();
    if num_elements == 0 {
        return Err("at least one element mass is required".to_string());
    }
    if input.real_masses.len() != num_elements
        || input.bounds.len() != num_elements
        || input.min_values.len() != num_elements
        || input.rdbe_coeffs.len() != num_elements
        || input.iso_m1_coeffs.len() != num_elements
        || input.iso_m2_direct.len() != num_elements
    {
        return Err("all per-element arrays must have the same length".to_string());
    }
    if input.max_results < 0 {
        return Err("max_results must be non-negative".to_string());
    }
    let first_mass = input.integer_masses[0];
    if first_mass <= 0 {
        return Err("first integer mass must be positive".to_string());
    }
    if input.ert.len() != first_mass as usize {
        return Err("ERT row count must match the first integer mass".to_string());
    }
    for (row_idx, row) in input.ert.iter().enumerate() {
        if row.len() != num_elements {
            return Err(format!(
                "ERT row {} has length {}, expected {}",
                row_idx,
                row.len(),
                num_elements
            ));
        }
    }
    Ok(())
}

fn bound_to_i64(bound: f64) -> i64 {
    if bound.is_infinite() && bound.is_sign_positive() {
        i64::MAX
    } else if bound.is_infinite() && bound.is_sign_negative() {
        i64::MIN
    } else if bound.is_nan() {
        0
    } else {
        bound as i64
    }
}

fn is_decomposable(ert: &[Vec<f64>], i: usize, m: i64, a1: i64) -> bool {
    if m < 0 {
        return false;
    }
    let residue = (m % a1) as usize;
    ert[residue][i] <= m as f64
}

fn decompose_counts(input: &DecomposeInput) -> Vec<Vec<i32>> {
    let num_elements = input.integer_masses.len();
    if input.max_results == 0 || input.max_int < input.min_int {
        return Vec::new();
    }

    let a1 = input.integer_masses[0];
    let k = num_elements - 1;
    let bounds_i: Vec<i64> = input.bounds.iter().map(|v| bound_to_i64(*v)).collect();
    let max_results = input.max_results as usize;

    let mut out_counts: Vec<Vec<i32>> = Vec::new();
    let mut mass_valid_count = 0_usize;

    for m_target in input.min_int..=input.max_int {
        let mut c = vec![0_i64; num_elements];
        let mut i = k;
        let mut m = m_target;

        while i <= k && mass_valid_count < max_results {
            if !is_decomposable(&input.ert, i, m, a1) {
                while i <= k && !is_decomposable(&input.ert, i, m, a1) {
                    m += c[i] * input.integer_masses[i];
                    c[i] = 0;
                    i += 1;
                }
                while i <= k && c[i] >= bounds_i[i] {
                    m += c[i] * input.integer_masses[i];
                    c[i] = 0;
                    i += 1;
                }
                if i <= k {
                    m -= input.integer_masses[i];
                    c[i] += 1;
                }
            } else {
                while i > 0 && is_decomposable(&input.ert, i - 1, m, a1) {
                    i -= 1;
                }

                if i == 0 {
                    c[0] = m / a1;

                    if c[0] <= bounds_i[0] {
                        let mut total = 0_i64;
                        let mut exact_mass = -input.charge_mass_offset;
                        let mut approx_m1 = 0.0;
                        let mut full_counts = vec![0_i64; num_elements];

                        for j in 0..num_elements {
                            let val = c[j] + input.min_values[j];
                            full_counts[j] = val;
                            total += val;
                            exact_mass += (val as f64) * input.real_masses[j];
                            approx_m1 += (val as f64) * input.iso_m1_coeffs[j];
                        }

                        if total > 0
                            && input.original_min_mass <= exact_mass
                            && exact_mass <= input.original_max_mass
                        {
                            mass_valid_count += 1;
                            let mut store = true;

                            if input.do_rdbe_filter {
                                let rdbe = chemistry::rdbe_from_counts_i64(
                                    &full_counts,
                                    &input.rdbe_coeffs,
                                );
                                store = filters::passes_rdbe_and_octet(
                                    rdbe,
                                    input.rdbe_min,
                                    input.rdbe_max,
                                    input.check_octet,
                                    input.charge_parity_even,
                                );
                            }

                            if store && input.do_iso_filter {
                                store = isotope::within_ratio_tolerance(
                                    approx_m1,
                                    input.obs_m1_ratio,
                                    input.iso_tol_rel,
                                    input.iso_tol_abs,
                                );

                                if store && input.obs_m2_ratio > 0.0 {
                                    let approx_m2 = isotope::approx_m2_from_counts(
                                        approx_m1,
                                        &full_counts,
                                        &input.iso_m2_direct,
                                    );
                                    store = isotope::within_ratio_tolerance(
                                        approx_m2,
                                        input.obs_m2_ratio,
                                        input.iso_tol_rel,
                                        input.iso_tol_abs,
                                    );
                                }
                            }

                            if store {
                                out_counts.push(full_counts.iter().map(|v| *v as i32).collect());
                            }
                        }
                    }

                    i += 1;
                }

                while i <= k && c[i] >= bounds_i[i] {
                    m += c[i] * input.integer_masses[i];
                    c[i] = 0;
                    i += 1;
                }
                if i <= k {
                    m -= input.integer_masses[i];
                    c[i] += 1;
                }
            }
        }
    }

    out_counts
}

pub fn decompose_and_score(input: &DecomposeInput) -> Result<DecomposeOutput, String> {
    validate_input(input)?;

    let counts = decompose_counts(&input);
    let n_results = counts.len();
    if n_results == 0 {
        return Ok(DecomposeOutput {
            counts,
            exact_masses: Vec::new(),
            error_ppm: Vec::new(),
            error_da: Vec::new(),
            rdbe: if input.compute_rdbe {
                Some(Vec::new())
            } else {
                None
            },
        });
    }

    let mut exact_masses = Vec::with_capacity(n_results);
    let mut error_ppm = Vec::with_capacity(n_results);
    let mut error_da = Vec::with_capacity(n_results);
    let mut abs_err = Vec::with_capacity(n_results);
    let mut rdbe = if input.compute_rdbe {
        Some(Vec::with_capacity(n_results))
    } else {
        None
    };

    for row in &counts {
        let mut exact_mass = -input.charge_mass_offset;
        for (count, mass) in row.iter().zip(input.real_masses.iter()) {
            exact_mass += (*count as f64) * mass;
        }
        exact_mass += input.adduct_mass;

        let da = exact_mass - input.query_mass;
        let ppm = da / input.query_mass * 1e6;

        exact_masses.push(exact_mass);
        error_da.push(da);
        error_ppm.push(ppm);
        abs_err.push(ppm.abs());

        if let Some(rdbe_values) = rdbe.as_mut() {
            rdbe_values.push(chemistry::rdbe_from_counts_i32(row, &input.rdbe_coeffs));
        }
    }

    let mut order: Vec<usize> = (0..n_results).collect();
    order.sort_by(|a, b| {
        abs_err[*a]
            .partial_cmp(&abs_err[*b])
            .unwrap_or(Ordering::Greater)
            .then_with(|| a.cmp(b))
    });

    let mut sorted_counts = Vec::with_capacity(n_results);
    let mut sorted_masses = Vec::with_capacity(n_results);
    let mut sorted_ppm = Vec::with_capacity(n_results);
    let mut sorted_da = Vec::with_capacity(n_results);
    let mut sorted_rdbe = rdbe.as_ref().map(|_| Vec::with_capacity(n_results));

    for idx in order {
        sorted_counts.push(counts[idx].clone());
        sorted_masses.push(exact_masses[idx]);
        sorted_ppm.push(error_ppm[idx]);
        sorted_da.push(error_da[idx]);
        if let (Some(values), Some(out)) = (rdbe.as_ref(), sorted_rdbe.as_mut()) {
            out.push(values[idx]);
        }
    }

    Ok(DecomposeOutput {
        counts: sorted_counts,
        exact_masses: sorted_masses,
        error_ppm: sorted_ppm,
        error_da: sorted_da,
        rdbe: sorted_rdbe,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_decomposer_from_masses, decompose_and_score, DecomposeInput};

    fn two_element_input() -> DecomposeInput {
        DecomposeInput {
            ert: std::sync::Arc::new(vec![vec![0.0, 0.0]]),
            integer_masses: std::sync::Arc::new(vec![1, 2]),
            real_masses: std::sync::Arc::new(vec![1.0, 2.0]),
            bounds: vec![f64::INFINITY, f64::INFINITY],
            min_values: vec![0, 0],
            min_int: 2,
            max_int: 2,
            original_min_mass: 2.0,
            original_max_mass: 2.0,
            charge_mass_offset: 0.0,
            max_results: 10,
            rdbe_coeffs: vec![0.0, 0.0],
            rdbe_min: f64::NEG_INFINITY,
            rdbe_max: f64::INFINITY,
            check_octet: false,
            charge_parity_even: true,
            do_rdbe_filter: false,
            do_iso_filter: false,
            iso_m1_coeffs: vec![0.0, 0.0],
            iso_m2_direct: vec![0.0, 0.0],
            obs_m1_ratio: 0.0,
            obs_m2_ratio: 0.0,
            iso_tol_rel: 0.3,
            iso_tol_abs: 0.02,
            query_mass: 2.0,
            adduct_mass: 0.0,
            compute_rdbe: false,
        }
    }

    #[test]
    fn decomposes_and_scores_simple_integer_system() {
        let input = two_element_input();
        let output = decompose_and_score(&input).unwrap();
        assert_eq!(output.counts, vec![vec![2, 0], vec![0, 1]]);
        assert_eq!(output.exact_masses, vec![2.0, 2.0]);
        assert_eq!(output.error_da, vec![0.0, 0.0]);
    }

    #[test]
    fn max_results_zero_returns_empty_output() {
        let mut input = two_element_input();
        input.max_results = 0;
        let output = decompose_and_score(&input).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn builds_discretized_ert_from_unsorted_masses() {
        let built =
            build_decomposer_from_masses(vec!["B".to_string(), "A".to_string()], vec![2.0, 1.0])
                .unwrap();

        assert_eq!(built.element_symbols, vec!["A", "B"]);
        assert_eq!(built.real_masses, vec![1.0, 2.0]);
        assert_eq!(built.integer_masses, vec![1, 2]);
        assert_eq!(built.ert, vec![vec![0.0, 0.0]]);
        assert!((built.precision - 0.9999433728194304).abs() < 1e-15);
        assert!(built.min_error < 0.0);
        assert_eq!(built.max_error, 0.0);
    }
}
