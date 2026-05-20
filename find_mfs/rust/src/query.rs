use std::collections::HashMap;

use crate::decomposer::{self, CountTable, DecomposeInput, DecomposeOutput};
use crate::filters;
#[cfg(test)]
use crate::formula;
use crate::isospec_ffi::{self, IsotopeScoreOutput, IsotopeScoringInput};

pub struct FindFormulaeInput {
    pub decompose: DecomposeInput,
    pub core_symbols: Vec<String>,
    pub charge: i32,
    pub remaining_apply_rdbe_filter: bool,
    pub remaining_rdbe_min: f64,
    pub remaining_rdbe_max: f64,
    pub remaining_check_octet: bool,
    pub can_compute_rdbe: bool,
    pub adduct_present: bool,
    pub adduct_symbols: Vec<String>,
    pub adduct_counts: Vec<i32>,
    pub unknown_symbol_indices: Vec<usize>,
    pub isotope: Option<IsotopeQueryInput>,
}

pub struct IsotopeQueryInput {
    pub symbols: Vec<String>,
    pub scoring: IsotopeScoringInput,
    pub minimum_rmse: f64,
}

#[derive(Clone)]
pub struct FindFormulaeOutput {
    pub core_symbols: Vec<String>,
    pub formula_charge: i32,
    pub rdbe_coeffs: Vec<f64>,
    pub counts: CountTable,
    pub exact_masses: Vec<f64>,
    pub error_ppm: Vec<f64>,
    pub error_da: Vec<f64>,
    pub rdbe: Option<Vec<f64>>,
    pub iso_rmse: Option<Vec<f64>>,
    pub iso_match_frac: Option<Vec<f64>>,
    pub iso_n_matched: Option<Vec<i32>>,
    pub iso_peak_matches: Option<Vec<Vec<i8>>>,
    pub formula_strings: Option<Vec<String>>,
}

pub fn find_formulae(input: FindFormulaeInput) -> Result<FindFormulaeOutput, String> {
    let rdbe_coeffs = if input.can_compute_rdbe {
        input.decompose.rdbe_coeffs.clone()
    } else {
        Vec::new()
    };
    let mut raw = decomposer::decompose_and_score(&input.decompose)?;

    apply_residual_filters(&mut raw, &input)?;

    let mut isotope_output = None;
    if let Some(isotope) = input.isotope.as_ref() {
        isotope_output = Some(apply_isotope_filter(&mut raw, &input, isotope)?);
    }

    let formula_charge = if input.adduct_present {
        0
    } else {
        input.charge
    };

    Ok(FindFormulaeOutput {
        core_symbols: input.core_symbols,
        formula_charge,
        rdbe_coeffs,
        formula_strings: None,
        counts: raw.counts,
        exact_masses: raw.exact_masses,
        error_ppm: raw.error_ppm,
        error_da: raw.error_da,
        rdbe: raw.rdbe,
        iso_rmse: isotope_output.as_ref().map(|out| out.rmse.clone()),
        iso_match_frac: isotope_output
            .as_ref()
            .map(|out| out.match_fraction.clone()),
        iso_n_matched: isotope_output.as_ref().map(|out| out.n_matched.clone()),
        iso_peak_matches: isotope_output.map(|out| out.peak_matches),
    })
}

#[cfg(test)]
fn format_core_formula_strings(
    core_symbols: &[String],
    counts: &[Vec<i32>],
    charge: i32,
) -> Vec<String> {
    counts
        .iter()
        .map(|row| {
            let row_i64: Vec<i64> = row.iter().map(|count| *count as i64).collect();
            formula::format_formula_from_counts(core_symbols, &row_i64, charge)
        })
        .collect()
}

fn apply_residual_filters(
    raw: &mut DecomposeOutput,
    input: &FindFormulaeInput,
) -> Result<(), String> {
    if raw.is_empty()
        || (!input.remaining_apply_rdbe_filter
            && !input.remaining_check_octet
            && input.unknown_symbol_indices.is_empty())
    {
        return Ok(());
    }

    let Some(rdbe) = raw.rdbe.as_ref() else {
        raw.retain_by_mask(&vec![false; raw.len()]);
        return Ok(());
    };

    let core_charge = if input.adduct_present {
        0
    } else {
        input.charge
    };
    let charge_parity_even = core_charge.abs() % 2 == 0;
    let mut mask = Vec::with_capacity(raw.len());

    for (idx, row) in raw.counts.rows().enumerate() {
        let mut keep = true;
        if !input.unknown_symbol_indices.is_empty() {
            keep = input
                .unknown_symbol_indices
                .iter()
                .all(|col| row.get(*col).copied().unwrap_or(0) == 0);
        }

        if keep && input.remaining_apply_rdbe_filter {
            keep = rdbe[idx] >= input.remaining_rdbe_min && rdbe[idx] <= input.remaining_rdbe_max;
        }

        if keep && input.remaining_check_octet {
            keep = filters::passes_residual_octet(rdbe[idx], charge_parity_even);
        }

        mask.push(keep);
    }

    raw.retain_by_mask(&mask);
    Ok(())
}

fn apply_isotope_filter(
    raw: &mut DecomposeOutput,
    input: &FindFormulaeInput,
    isotope: &IsotopeQueryInput,
) -> Result<IsotopeScoreOutput, String> {
    if raw.is_empty() {
        return Ok(IsotopeScoreOutput {
            rmse: Vec::new(),
            match_fraction: Vec::new(),
            n_matched: Vec::new(),
            peak_matches: Vec::new(),
        });
    }

    let ion_counts = build_ion_counts(raw, input, &isotope.symbols)?;
    let mut nonnegative_mask = Vec::with_capacity(ion_counts.len());
    for row in &ion_counts {
        nonnegative_mask.push(row.iter().all(|count| *count >= 0));
    }

    let filtered_ion_counts = if nonnegative_mask.iter().all(|keep| *keep) {
        ion_counts
    } else {
        raw.retain_by_mask(&nonnegative_mask);
        ion_counts
            .into_iter()
            .zip(nonnegative_mask.iter())
            .filter_map(|(row, keep)| if *keep { Some(row) } else { None })
            .collect()
    };

    if raw.is_empty() {
        return Ok(IsotopeScoreOutput {
            rmse: Vec::new(),
            match_fraction: Vec::new(),
            n_matched: Vec::new(),
            peak_matches: Vec::new(),
        });
    }

    let scored = isospec_ffi::score_isotope_batch(&filtered_ion_counts, &isotope.scoring)?;
    let keep_mask: Vec<bool> = scored
        .rmse
        .iter()
        .map(|rmse| *rmse <= isotope.minimum_rmse)
        .collect();

    if keep_mask.iter().all(|keep| *keep) {
        return Ok(scored);
    }

    raw.retain_by_mask(&keep_mask);
    Ok(IsotopeScoreOutput {
        rmse: decomposer::retain_vec_by_mask(&scored.rmse, &keep_mask),
        match_fraction: decomposer::retain_vec_by_mask(&scored.match_fraction, &keep_mask),
        n_matched: decomposer::retain_vec_by_mask(&scored.n_matched, &keep_mask),
        peak_matches: decomposer::retain_vec_by_mask(&scored.peak_matches, &keep_mask),
    })
}

fn build_ion_counts(
    raw: &DecomposeOutput,
    input: &FindFormulaeInput,
    ion_symbols: &[String],
) -> Result<Vec<Vec<i32>>, String> {
    let ion_index: HashMap<&str, usize> = ion_symbols
        .iter()
        .enumerate()
        .map(|(idx, symbol)| (symbol.as_str(), idx))
        .collect();

    let mut core_to_ion = Vec::with_capacity(input.core_symbols.len());
    for symbol in &input.core_symbols {
        let Some(idx) = ion_index.get(symbol.as_str()) else {
            return Err(format!(
                "core symbol '{symbol}' missing from isotope symbol list"
            ));
        };
        core_to_ion.push(*idx);
    }

    let mut offsets = vec![0_i32; ion_symbols.len()];
    for (symbol, count) in input.adduct_symbols.iter().zip(input.adduct_counts.iter()) {
        let Some(idx) = ion_index.get(symbol.as_str()) else {
            return Err(format!(
                "adduct symbol '{symbol}' missing from isotope symbol list"
            ));
        };
        offsets[*idx] += *count;
    }

    let mut ion_counts = Vec::with_capacity(raw.counts.len());
    for core_row in raw.counts.rows() {
        let mut row = offsets.clone();
        for (core_idx, count) in core_row.iter().enumerate() {
            row[core_to_ion[core_idx]] += *count;
        }
        ion_counts.push(row);
    }

    Ok(ion_counts)
}

#[cfg(test)]
mod tests {
    use super::{apply_residual_filters, format_core_formula_strings, FindFormulaeInput};
    use crate::decomposer::{CountTable, DecomposeInput, DecomposeOutput};

    fn empty_decompose_input() -> DecomposeInput {
        DecomposeInput {
            ert: std::sync::Arc::new(vec![0.0]),
            integer_masses: std::sync::Arc::new(vec![1]),
            real_masses: std::sync::Arc::new(vec![1.0]),
            bounds: vec![f64::INFINITY],
            min_values: vec![0],
            min_int: 1,
            max_int: 1,
            original_min_mass: 1.0,
            original_max_mass: 1.0,
            charge_mass_offset: 0.0,
            max_results: 10,
            rdbe_coeffs: vec![0.0],
            rdbe_min: f64::NEG_INFINITY,
            rdbe_max: f64::INFINITY,
            check_octet: false,
            charge_parity_even: true,
            do_rdbe_filter: false,
            do_iso_filter: false,
            iso_m1_coeffs: vec![0.0],
            iso_m2_direct: vec![0.0],
            obs_m1_ratio: 0.0,
            obs_m2_ratio: 0.0,
            iso_tol_rel: 0.3,
            iso_tol_abs: 0.02,
            query_mass: 1.0,
            adduct_mass: 0.0,
            compute_rdbe: true,
        }
    }

    fn input() -> FindFormulaeInput {
        FindFormulaeInput {
            decompose: empty_decompose_input(),
            core_symbols: vec!["C".to_string(), "X".to_string()],
            charge: 0,
            remaining_apply_rdbe_filter: true,
            remaining_rdbe_min: 0.0,
            remaining_rdbe_max: 5.0,
            remaining_check_octet: false,
            can_compute_rdbe: true,
            adduct_present: false,
            adduct_symbols: Vec::new(),
            adduct_counts: Vec::new(),
            unknown_symbol_indices: vec![1],
            isotope: None,
        }
    }

    #[test]
    fn residual_filters_apply_unknown_symbol_and_rdbe_masks() {
        let mut raw = DecomposeOutput {
            counts: CountTable::from_rows(vec![vec![1, 0], vec![1, 1], vec![2, 0]]),
            exact_masses: vec![1.0, 2.0, 3.0],
            error_ppm: vec![0.0, 0.0, 0.0],
            error_da: vec![0.0, 0.0, 0.0],
            rdbe: Some(vec![1.0, 1.0, 6.0]),
        };

        apply_residual_filters(&mut raw, &input()).unwrap();
        assert_eq!(raw.counts.to_rows(), vec![vec![1, 0]]);
        assert_eq!(raw.exact_masses, vec![1.0]);
    }

    #[test]
    fn formula_strings_format_core_counts() {
        let symbols = vec!["C".to_string(), "H".to_string(), "O".to_string()];
        let counts = vec![vec![6, 12, 6], vec![1, 0, 2]];

        assert_eq!(
            format_core_formula_strings(&symbols, &counts, 1),
            vec!["[C6H12O6]+".to_string(), "[CO2]+".to_string()]
        );
        assert_eq!(
            format_core_formula_strings(&symbols, &counts[..1], 0),
            vec!["C6H12O6".to_string()]
        );
    }
}
