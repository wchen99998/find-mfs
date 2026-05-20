use std::collections::HashMap;
use std::sync::Arc;

use crate::chemistry;
use crate::decomposer::{self, DecomposeInput};
use crate::formula::{self, ParsedAdduct};
use crate::isospec_ffi::{self, IsotopeScoringInput};
use crate::query::{self, FindFormulaeInput, FindFormulaeOutput, IsotopeQueryInput};
use crate::static_data;

#[derive(Clone)]
pub struct StoredFormulaFinder {
    pub element_symbols: Vec<String>,
    pub ert: Arc<Vec<i64>>,
    pub integer_masses: Arc<Vec<i64>>,
    pub real_masses: Arc<Vec<f64>>,
    pub precision: f64,
    pub min_error: f64,
    pub max_error: f64,
    pub element_masses: HashMap<String, f64>,
    pub rdbe_coeffs_fallback: Vec<f64>,
    pub has_known_bond_electrons: bool,
    pub unknown_bond_electron_indices: Vec<usize>,
    pub iso_m1_coeffs: Vec<f64>,
    pub iso_m2_direct: Vec<f64>,
    pub isotope_table: StoredIsotopeTable,
}

#[derive(Clone)]
pub struct StoredIsotopeTable {
    pub lib_path: String,
    pub isotope_numbers: Vec<i32>,
    pub flat_masses: Vec<f64>,
    pub flat_probs: Vec<f64>,
    index_by_symbol: HashMap<String, usize>,
    offsets: Vec<usize>,
}

impl StoredIsotopeTable {
    pub fn new(
        lib_path: String,
        symbols: Vec<String>,
        isotope_numbers: Vec<i32>,
        flat_masses: Vec<f64>,
        flat_probs: Vec<f64>,
    ) -> Result<Self, String> {
        if symbols.len() != isotope_numbers.len() {
            return Err("isotope symbols and isotope number arrays must match".to_string());
        }
        if flat_masses.len() != flat_probs.len() {
            return Err("flat isotope mass/probability arrays must match".to_string());
        }

        let mut offsets = Vec::with_capacity(isotope_numbers.len() + 1);
        offsets.push(0);
        for isotope_number in &isotope_numbers {
            if *isotope_number < 0 {
                return Err("isotope numbers must be non-negative".to_string());
            }
            let next = offsets.last().copied().unwrap_or(0) + (*isotope_number as usize);
            offsets.push(next);
        }
        if offsets.last().copied().unwrap_or(0) != flat_masses.len() {
            return Err(format!(
                "flat isotope arrays have length {}, expected {}",
                flat_masses.len(),
                offsets.last().copied().unwrap_or(0)
            ));
        }

        let mut index_by_symbol = HashMap::with_capacity(symbols.len());
        for (idx, symbol) in symbols.iter().enumerate() {
            if index_by_symbol.insert(symbol.clone(), idx).is_some() {
                return Err(format!("duplicate isotope table symbol '{symbol}'"));
            }
        }

        Ok(Self {
            lib_path,
            isotope_numbers,
            flat_masses,
            flat_probs,
            index_by_symbol,
            offsets,
        })
    }

    pub fn build_scoring_input(
        &self,
        symbols: Vec<String>,
        observed_mz: Vec<f64>,
        observed_intensity: Vec<f64>,
        mz_match_tolerance: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
        charge: i32,
        electron_mass: f64,
    ) -> Result<IsotopeScoringInput, String> {
        let mut isotope_numbers = Vec::with_capacity(symbols.len());
        let mut flat_masses = Vec::new();
        let mut flat_probs = Vec::new();

        for symbol in &symbols {
            let Some(idx) = self.index_by_symbol.get(symbol.as_str()).copied() else {
                return Err(format!("symbol '{symbol}' missing from isotope table"));
            };
            isotope_numbers.push(self.isotope_numbers[idx]);
            let start = self.offsets[idx];
            let end = self.offsets[idx + 1];
            flat_masses.extend_from_slice(&self.flat_masses[start..end]);
            flat_probs.extend_from_slice(&self.flat_probs[start..end]);
        }

        Ok(IsotopeScoringInput {
            lib_path: self.lib_path.clone(),
            isotope_numbers,
            flat_masses,
            flat_probs,
            observed_mz,
            observed_intensity,
            mz_match_tolerance,
            simulated_mz_tolerance,
            simulated_intensity_threshold,
            charge,
            electron_mass,
        })
    }
}

#[derive(Clone)]
struct ElementSourceTable {
    element_masses: HashMap<String, f64>,
    isotope_coeffs: HashMap<String, (f64, f64)>,
}

impl ElementSourceTable {
    fn new(
        symbols: Vec<String>,
        isotope_numbers: Vec<i32>,
        flat_mass_numbers: Vec<i32>,
        flat_masses: Vec<f64>,
        flat_abundances: Vec<f64>,
    ) -> Result<Self, String> {
        if symbols.len() != isotope_numbers.len() {
            return Err("element source symbols and isotope counts must match".to_string());
        }
        if flat_mass_numbers.len() != flat_masses.len()
            || flat_mass_numbers.len() != flat_abundances.len()
        {
            return Err("flat element isotope source arrays must match".to_string());
        }

        let mut offsets = Vec::with_capacity(isotope_numbers.len() + 1);
        offsets.push(0);
        for isotope_number in &isotope_numbers {
            if *isotope_number <= 0 {
                return Err("element isotope counts must be positive".to_string());
            }
            let next = offsets.last().copied().unwrap_or(0) + (*isotope_number as usize);
            offsets.push(next);
        }
        if offsets.last().copied().unwrap_or(0) != flat_mass_numbers.len() {
            return Err(format!(
                "flat element isotope source arrays have length {}, expected {}",
                flat_mass_numbers.len(),
                offsets.last().copied().unwrap_or(0)
            ));
        }

        let mut element_masses = HashMap::with_capacity(symbols.len());
        let mut isotope_coeffs = HashMap::with_capacity(symbols.len());

        for (idx, symbol) in symbols.into_iter().enumerate() {
            let start = offsets[idx];
            let end = offsets[idx + 1];
            if element_masses.contains_key(symbol.as_str()) {
                return Err(format!("duplicate element source symbol '{symbol}'"));
            }

            let mut most_abundant_mass = flat_masses[start];
            let mut max_abundance = flat_abundances[start];
            let mut mono_mass_number = flat_mass_numbers[start];
            let mut mono_abundance = flat_abundances[start];

            for isotope_idx in start..end {
                let abundance = flat_abundances[isotope_idx];
                if abundance > max_abundance {
                    max_abundance = abundance;
                    most_abundant_mass = flat_masses[isotope_idx];
                }
                let mass_number = flat_mass_numbers[isotope_idx];
                if mass_number < mono_mass_number {
                    mono_mass_number = mass_number;
                    mono_abundance = abundance;
                }
            }
            if mono_abundance <= 0.0 {
                return Err(format!(
                    "monoisotopic abundance for element '{symbol}' must be positive"
                ));
            }

            let mut m1 = 0.0;
            let mut m2 = 0.0;
            for isotope_idx in start..end {
                match flat_mass_numbers[isotope_idx] - mono_mass_number {
                    1 => m1 += flat_abundances[isotope_idx] / mono_abundance,
                    2 => m2 += flat_abundances[isotope_idx] / mono_abundance,
                    _ => {}
                }
            }

            element_masses.insert(symbol.clone(), most_abundant_mass);
            isotope_coeffs.insert(symbol, (m1, m2));
        }

        Ok(Self {
            element_masses,
            isotope_coeffs,
        })
    }

    fn element_masses_for_symbols(&self, symbols: &[String]) -> Result<Vec<f64>, String> {
        symbols
            .iter()
            .map(|symbol| {
                self.element_masses
                    .get(symbol)
                    .copied()
                    .ok_or_else(|| format!("missing isotope source data for element '{symbol}'"))
            })
            .collect()
    }

    fn isotope_coeffs_for_symbols(
        &self,
        symbols: &[String],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut m1_values = Vec::with_capacity(symbols.len());
        let mut m2_values = Vec::with_capacity(symbols.len());
        for symbol in symbols {
            let Some((m1, m2)) = self.isotope_coeffs.get(symbol) else {
                return Err(format!(
                    "missing isotope prefilter coefficients for element '{symbol}'"
                ));
            };
            m1_values.push(*m1);
            m2_values.push(*m2);
        }
        Ok((m1_values, m2_values))
    }

    fn element_mass_vectors(&self) -> (Vec<String>, Vec<f64>) {
        let mut items: Vec<_> = self
            .element_masses
            .iter()
            .map(|(symbol, mass)| (symbol.clone(), *mass))
            .collect();
        items.sort_by(|left, right| left.0.cmp(&right.0));
        items.into_iter().unzip()
    }
}

#[allow(clippy::too_many_arguments)]
impl StoredFormulaFinder {
    pub fn new(
        element_symbols: Vec<String>,
        ert: Vec<Vec<f64>>,
        integer_masses: Vec<i64>,
        real_masses: Vec<f64>,
        precision: f64,
        min_error: f64,
        max_error: f64,
        element_mass_symbols: Vec<String>,
        element_mass_values: Vec<f64>,
        iso_m1_coeffs: Vec<f64>,
        iso_m2_direct: Vec<f64>,
        isospec_lib_path: String,
        isotope_table_symbols: Vec<String>,
        isotope_numbers: Vec<i32>,
        flat_isotope_masses: Vec<f64>,
        flat_isotope_probs: Vec<f64>,
    ) -> Result<Self, String> {
        let n_elements = element_symbols.len();
        if n_elements == 0 {
            return Err("at least one element is required".to_string());
        }
        if integer_masses.len() != n_elements || real_masses.len() != n_elements {
            return Err("element symbols and mass arrays must have the same length".to_string());
        }
        if ert.len() != integer_masses[0] as usize {
            return Err("ERT row count must match first integer mass".to_string());
        }
        let mut flat_ert = Vec::with_capacity(ert.len() * n_elements);
        for (row_idx, row) in ert.iter().enumerate() {
            if row.len() != n_elements {
                return Err(format!(
                    "ERT row {row_idx} has length {}, expected {n_elements}",
                    row.len()
                ));
            }
            flat_ert.extend(row.iter().map(|value| {
                if value.is_infinite() && value.is_sign_positive() {
                    i64::MAX
                } else {
                    *value as i64
                }
            }));
        }
        if element_mass_symbols.len() != element_mass_values.len() {
            return Err("element mass symbol/value arrays must have the same length".to_string());
        }
        if iso_m1_coeffs.len() != n_elements || iso_m2_direct.len() != n_elements {
            return Err("isotope coefficient arrays must match element count".to_string());
        }
        let mut element_masses = HashMap::with_capacity(element_mass_symbols.len());
        for (symbol, mass) in element_mass_symbols
            .into_iter()
            .zip(element_mass_values.into_iter())
        {
            element_masses.insert(symbol, mass);
        }
        let rdbe_coeffs_fallback: Vec<f64> = element_symbols
            .iter()
            .map(|symbol| chemistry::rdbe_coeff_for_symbol(symbol))
            .collect();
        let unknown_bond_electron_indices: Vec<usize> = element_symbols
            .iter()
            .enumerate()
            .filter_map(|(idx, symbol)| {
                if chemistry::bond_electrons(symbol).is_none() {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        let has_known_bond_electrons = unknown_bond_electron_indices.is_empty();
        let isotope_table = StoredIsotopeTable::new(
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )?;

        Ok(Self {
            element_symbols,
            ert: Arc::new(flat_ert),
            integer_masses: Arc::new(integer_masses),
            real_masses: Arc::new(real_masses),
            precision,
            min_error,
            max_error,
            element_masses,
            rdbe_coeffs_fallback,
            has_known_bond_electrons,
            unknown_bond_electron_indices,
            iso_m1_coeffs,
            iso_m2_direct,
            isotope_table,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_element_masses(
        element_symbols: Vec<String>,
        element_masses_for_finder: Vec<f64>,
        element_mass_symbols: Vec<String>,
        element_mass_values: Vec<f64>,
        isotope_coeff_symbols: Vec<String>,
        isotope_m1_values: Vec<f64>,
        isotope_m2_values: Vec<f64>,
        isospec_lib_path: String,
        isotope_table_symbols: Vec<String>,
        isotope_numbers: Vec<i32>,
        flat_isotope_masses: Vec<f64>,
        flat_isotope_probs: Vec<f64>,
    ) -> Result<Self, String> {
        let built =
            decomposer::build_decomposer_from_masses(element_symbols, element_masses_for_finder)?;
        if isotope_coeff_symbols.len() != isotope_m1_values.len()
            || isotope_coeff_symbols.len() != isotope_m2_values.len()
        {
            return Err("isotope coefficient symbol/value arrays must match".to_string());
        }
        let isotope_coeffs_by_symbol: HashMap<String, (f64, f64)> = isotope_coeff_symbols
            .into_iter()
            .zip(isotope_m1_values.into_iter().zip(isotope_m2_values))
            .collect();
        let mut iso_m1_coeffs = Vec::with_capacity(built.element_symbols.len());
        let mut iso_m2_direct = Vec::with_capacity(built.element_symbols.len());
        for symbol in &built.element_symbols {
            let Some((m1, m2)) = isotope_coeffs_by_symbol.get(symbol) else {
                return Err(format!(
                    "missing isotope prefilter coefficients for element '{symbol}'"
                ));
            };
            iso_m1_coeffs.push(*m1);
            iso_m2_direct.push(*m2);
        }

        Self::new(
            built.element_symbols,
            built.ert,
            built.integer_masses,
            built.real_masses,
            built.precision,
            built.min_error,
            built.max_error,
            element_mass_symbols,
            element_mass_values,
            iso_m1_coeffs,
            iso_m2_direct,
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_precomputed_sources(
        element_symbols: Vec<String>,
        ert: Vec<Vec<f64>>,
        integer_masses: Vec<i64>,
        real_masses: Vec<f64>,
        precision: f64,
        min_error: f64,
        max_error: f64,
        element_source_symbols: Vec<String>,
        element_source_isotope_numbers: Vec<i32>,
        flat_element_mass_numbers: Vec<i32>,
        flat_element_isotope_masses: Vec<f64>,
        flat_element_isotope_abundances: Vec<f64>,
        isospec_lib_path: String,
        isotope_table_symbols: Vec<String>,
        isotope_numbers: Vec<i32>,
        flat_isotope_masses: Vec<f64>,
        flat_isotope_probs: Vec<f64>,
    ) -> Result<Self, String> {
        let source_table = ElementSourceTable::new(
            element_source_symbols,
            element_source_isotope_numbers,
            flat_element_mass_numbers,
            flat_element_isotope_masses,
            flat_element_isotope_abundances,
        )?;
        let (iso_m1_coeffs, iso_m2_direct) =
            source_table.isotope_coeffs_for_symbols(&element_symbols)?;
        let (element_mass_symbols, element_mass_values) = source_table.element_mass_vectors();

        Self::new(
            element_symbols,
            ert,
            integer_masses,
            real_masses,
            precision,
            min_error,
            max_error,
            element_mass_symbols,
            element_mass_values,
            iso_m1_coeffs,
            iso_m2_direct,
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_element_sources(
        element_symbols: Vec<String>,
        element_source_symbols: Vec<String>,
        element_source_isotope_numbers: Vec<i32>,
        flat_element_mass_numbers: Vec<i32>,
        flat_element_isotope_masses: Vec<f64>,
        flat_element_isotope_abundances: Vec<f64>,
        isospec_lib_path: String,
        isotope_table_symbols: Vec<String>,
        isotope_numbers: Vec<i32>,
        flat_isotope_masses: Vec<f64>,
        flat_isotope_probs: Vec<f64>,
    ) -> Result<Self, String> {
        let source_table = ElementSourceTable::new(
            element_source_symbols,
            element_source_isotope_numbers,
            flat_element_mass_numbers,
            flat_element_isotope_masses,
            flat_element_isotope_abundances,
        )?;
        let element_masses_for_finder =
            source_table.element_masses_for_symbols(&element_symbols)?;
        let built =
            decomposer::build_decomposer_from_masses(element_symbols, element_masses_for_finder)?;
        let (iso_m1_coeffs, iso_m2_direct) =
            source_table.isotope_coeffs_for_symbols(&built.element_symbols)?;
        let (element_mass_symbols, element_mass_values) = source_table.element_mass_vectors();

        Self::new(
            built.element_symbols,
            built.ert,
            built.integer_masses,
            built.real_masses,
            built.precision,
            built.min_error,
            built.max_error,
            element_mass_symbols,
            element_mass_values,
            iso_m1_coeffs,
            iso_m2_direct,
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_precomputed_embedded_sources(
        element_symbols: Vec<String>,
        ert: Vec<Vec<f64>>,
        integer_masses: Vec<i64>,
        real_masses: Vec<f64>,
        precision: f64,
        min_error: f64,
        max_error: f64,
        isospec_lib_path: String,
    ) -> Result<Self, String> {
        Self::from_precomputed_sources(
            element_symbols,
            ert,
            integer_masses,
            real_masses,
            precision,
            min_error,
            max_error,
            static_data::element_source_symbols(),
            static_data::ELEMENT_SOURCE_ISOTOPE_NUMBERS.to_vec(),
            static_data::FLAT_ELEMENT_MASS_NUMBERS.to_vec(),
            static_data::FLAT_ELEMENT_ISOTOPE_MASSES.to_vec(),
            static_data::FLAT_ELEMENT_ISOTOPE_ABUNDANCES.to_vec(),
            isospec_lib_path,
            static_data::isospec_table_symbols(),
            static_data::ISOTOPE_NUMBERS.to_vec(),
            static_data::FLAT_ISOTOPE_MASSES.to_vec(),
            static_data::FLAT_ISOTOPE_PROBS.to_vec(),
        )
    }

    pub fn from_embedded_sources(
        element_symbols: Vec<String>,
        isospec_lib_path: String,
    ) -> Result<Self, String> {
        Self::from_element_sources(
            element_symbols,
            static_data::element_source_symbols(),
            static_data::ELEMENT_SOURCE_ISOTOPE_NUMBERS.to_vec(),
            static_data::FLAT_ELEMENT_MASS_NUMBERS.to_vec(),
            static_data::FLAT_ELEMENT_ISOTOPE_MASSES.to_vec(),
            static_data::FLAT_ELEMENT_ISOTOPE_ABUNDANCES.to_vec(),
            isospec_lib_path,
            static_data::isospec_table_symbols(),
            static_data::ISOTOPE_NUMBERS.to_vec(),
            static_data::FLAT_ISOTOPE_MASSES.to_vec(),
            static_data::FLAT_ISOTOPE_PROBS.to_vec(),
        )
    }

    pub fn n_elements(&self) -> usize {
        self.element_symbols.len()
    }

    pub fn parse_min_counts(&self, formula: &str) -> Result<Vec<i64>, String> {
        formula::parse_formula_min_bounds(formula, &self.element_symbols)
    }

    pub fn parse_max_counts(&self, formula: &str) -> Result<Vec<f64>, String> {
        formula::parse_formula_max_bounds(formula, &self.element_symbols)
    }

    pub fn parse_adduct(&self, adduct: &str) -> Result<ParsedAdduct, String> {
        formula::parse_adduct(adduct, &self.element_masses)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn find_formulae_public(
        &self,
        mass: f64,
        charge: i32,
        ppm_error: f64,
        mz_error: f64,
        min_count_formula: Option<String>,
        min_count_symbols: Vec<String>,
        min_count_values: Vec<f64>,
        max_count_formula: Option<String>,
        max_count_symbols: Vec<String>,
        max_count_values: Vec<f64>,
        max_results: i32,
        filter_rdbe: Option<(f64, f64)>,
        check_octet: bool,
        adduct: Option<String>,
        enable_iso_prefilter: bool,
        observed_mz_for_prefilter: Vec<f64>,
        observed_intensity_for_prefilter: Vec<f64>,
        iso_tol_rel: f64,
        iso_tol_abs: f64,
        electron_mass: f64,
        do_isotope_match: bool,
        observed_mz: Vec<f64>,
        observed_intensity: Vec<f64>,
        mz_match_tolerance: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
        minimum_rmse: f64,
    ) -> Result<PublicFindFormulaeOutput, String> {
        let min_counts =
            self.build_min_count_vector(min_count_formula, &min_count_symbols, &min_count_values)?;
        let max_counts =
            self.build_max_count_vector(max_count_formula, &max_count_symbols, &max_count_values)?;

        let parsed_adduct = match adduct.as_deref() {
            Some(adduct) if !adduct.is_empty() => Some(self.parse_adduct(adduct)?),
            _ => None,
        };
        let adduct_mass = parsed_adduct
            .as_ref()
            .map(|parsed| parsed.mass)
            .unwrap_or(0.0);
        let adduct_symbols = parsed_adduct
            .as_ref()
            .map(|parsed| parsed.symbols.clone())
            .unwrap_or_default();
        let adduct_counts = parsed_adduct
            .as_ref()
            .map(|parsed| parsed.counts.clone())
            .unwrap_or_default();
        let adjusted_mass = mass - adduct_mass;
        let (apply_rdbe_filter, rdbe_min, rdbe_max) = match filter_rdbe {
            Some((min, max)) => (true, min, max),
            None => (false, 0.0, 0.0),
        };

        let output = self.find_formulae_configured(
            adjusted_mass,
            charge,
            ppm_error,
            mz_error,
            min_counts,
            max_counts,
            max_results,
            apply_rdbe_filter,
            rdbe_min,
            rdbe_max,
            check_octet,
            enable_iso_prefilter,
            observed_mz_for_prefilter,
            observed_intensity_for_prefilter,
            iso_tol_rel,
            iso_tol_abs,
            mass,
            adduct_mass,
            electron_mass,
            parsed_adduct.is_some(),
            adduct_symbols.clone(),
            adduct_counts.clone(),
            do_isotope_match,
            observed_mz,
            observed_intensity,
            mz_match_tolerance,
            simulated_mz_tolerance,
            simulated_intensity_threshold,
            minimum_rmse,
        )?;

        Ok(PublicFindFormulaeOutput {
            output,
            adduct_mass,
            adduct_symbols,
            adduct_counts,
        })
    }

    fn build_min_count_vector(
        &self,
        formula: Option<String>,
        symbols: &[String],
        values: &[f64],
    ) -> Result<Vec<i64>, String> {
        if let Some(formula) = formula {
            return self.parse_min_counts(&formula);
        }
        if symbols.len() != values.len() {
            return Err("min count symbol/value arrays must have the same length".to_string());
        }

        let symbol_to_idx = self.symbol_index();
        let mut counts = vec![0_i64; self.n_elements()];
        for (symbol, value) in symbols.iter().zip(values.iter()) {
            let Some(idx) = symbol_to_idx.get(symbol.as_str()).copied() else {
                continue;
            };
            counts[idx] = if value.is_infinite() {
                0
            } else {
                *value as i64
            };
        }
        Ok(counts)
    }

    fn build_max_count_vector(
        &self,
        formula: Option<String>,
        symbols: &[String],
        values: &[f64],
    ) -> Result<Vec<f64>, String> {
        if let Some(formula) = formula {
            return self.parse_max_counts(&formula);
        }
        if symbols.len() != values.len() {
            return Err("max count symbol/value arrays must have the same length".to_string());
        }

        let symbol_to_idx = self.symbol_index();
        let mut counts = vec![f64::INFINITY; self.n_elements()];
        for (symbol, value) in symbols.iter().zip(values.iter()) {
            let Some(idx) = symbol_to_idx.get(symbol.as_str()).copied() else {
                continue;
            };
            counts[idx] = *value;
        }
        Ok(counts)
    }

    fn symbol_index(&self) -> HashMap<&str, usize> {
        self.element_symbols
            .iter()
            .enumerate()
            .map(|(idx, symbol)| (symbol.as_str(), idx))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn find_formulae_configured(
        &self,
        query_mass: f64,
        charge: i32,
        ppm_error: f64,
        mz_error: f64,
        min_counts: Vec<i64>,
        max_counts: Vec<f64>,
        max_results: i32,
        apply_rdbe_filter: bool,
        rdbe_min: f64,
        rdbe_max: f64,
        check_octet: bool,
        enable_iso_prefilter: bool,
        observed_mz_for_prefilter: Vec<f64>,
        observed_intensity_for_prefilter: Vec<f64>,
        iso_tol_rel: f64,
        iso_tol_abs: f64,
        ion_query_mass: f64,
        adduct_mass: f64,
        electron_mass: f64,
        adduct_present: bool,
        adduct_symbols: Vec<String>,
        adduct_counts: Vec<i32>,
        do_isotope_match: bool,
        observed_mz: Vec<f64>,
        observed_intensity: Vec<f64>,
        mz_match_tolerance: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
        minimum_rmse: f64,
    ) -> Result<FindFormulaeOutput, String> {
        let has_rdbe_request = apply_rdbe_filter || check_octet;
        let can_compute_rdbe = self.has_known_bond_electrons || has_rdbe_request;
        let can_prefilter_rdbe = has_rdbe_request && self.has_known_bond_electrons;
        let compute_rdbe = has_rdbe_request && !can_prefilter_rdbe;

        let rdbe_coeffs = if can_compute_rdbe {
            self.rdbe_coeffs_fallback.clone()
        } else {
            vec![0.0; self.n_elements()]
        };
        let decompose_rdbe_min = if can_prefilter_rdbe && apply_rdbe_filter {
            rdbe_min
        } else {
            f64::NEG_INFINITY
        };
        let decompose_rdbe_max = if can_prefilter_rdbe && apply_rdbe_filter {
            rdbe_max
        } else {
            f64::INFINITY
        };
        let decompose_check_octet = can_prefilter_rdbe && check_octet;
        let core_charge_parity_even = if adduct_present {
            true
        } else {
            charge.abs() % 2 == 0
        };

        let (obs_m1_ratio, obs_m2_ratio) = if enable_iso_prefilter {
            observed_m1_m2_ratios(
                &observed_mz_for_prefilter,
                &observed_intensity_for_prefilter,
            )?
        } else {
            (0.0, 0.0)
        };
        let do_iso_filter = enable_iso_prefilter && obs_m1_ratio > 0.0;
        let (iso_m1_coeffs, iso_m2_direct) = if do_iso_filter {
            (self.iso_m1_coeffs.clone(), self.iso_m2_direct.clone())
        } else {
            (vec![0.0; self.n_elements()], vec![0.0; self.n_elements()])
        };

        let isotope = if do_isotope_match {
            Some(self.build_isotope_query_input(
                &adduct_symbols,
                charge,
                electron_mass,
                observed_mz,
                observed_intensity,
                mz_match_tolerance,
                simulated_mz_tolerance,
                simulated_intensity_threshold,
                minimum_rmse,
            )?)
        } else {
            None
        };

        self.find_formulae(
            query_mass,
            charge,
            ppm_error,
            mz_error,
            min_counts,
            max_counts,
            max_results,
            rdbe_coeffs,
            decompose_rdbe_min,
            decompose_rdbe_max,
            decompose_check_octet,
            Some(core_charge_parity_even),
            can_prefilter_rdbe,
            do_iso_filter,
            iso_m1_coeffs,
            iso_m2_direct,
            obs_m1_ratio,
            obs_m2_ratio,
            iso_tol_rel,
            iso_tol_abs,
            ion_query_mass,
            adduct_mass,
            compute_rdbe,
            can_compute_rdbe,
            electron_mass,
            !can_prefilter_rdbe && apply_rdbe_filter,
            rdbe_min,
            rdbe_max,
            !can_prefilter_rdbe && check_octet,
            adduct_present,
            adduct_symbols,
            adduct_counts,
            if !can_prefilter_rdbe && has_rdbe_request {
                self.unknown_bond_electron_indices.clone()
            } else {
                Vec::new()
            },
            isotope,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_isotope_query_input(
        &self,
        adduct_symbols: &[String],
        charge: i32,
        electron_mass: f64,
        observed_mz: Vec<f64>,
        observed_intensity: Vec<f64>,
        mz_match_tolerance: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
        minimum_rmse: f64,
    ) -> Result<StoredIsotopeQueryInput, String> {
        let mut symbols = self.element_symbols.clone();
        for symbol in adduct_symbols {
            if !symbols.iter().any(|existing| existing == symbol) {
                symbols.push(symbol.clone());
            }
        }

        let scoring = self.isotope_table.build_scoring_input(
            symbols.clone(),
            observed_mz,
            observed_intensity,
            mz_match_tolerance,
            simulated_mz_tolerance,
            simulated_intensity_threshold,
            charge,
            electron_mass,
        )?;

        Ok(StoredIsotopeQueryInput {
            symbols,
            scoring,
            minimum_rmse,
        })
    }

    pub fn simulate_isotope_envelope(
        &self,
        core_counts: Vec<i32>,
        adduct_symbols: Vec<String>,
        adduct_counts: Vec<i32>,
        charge: i32,
        electron_mass: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        if core_counts.len() != self.element_symbols.len() {
            return Err(format!(
                "core counts have length {}, expected {}",
                core_counts.len(),
                self.element_symbols.len()
            ));
        }
        if adduct_symbols.len() != adduct_counts.len() {
            return Err("adduct symbol/count arrays must match".to_string());
        }

        let mut symbols = self.element_symbols.clone();
        for symbol in &adduct_symbols {
            if !symbols.iter().any(|existing| existing == symbol) {
                symbols.push(symbol.clone());
            }
        }

        let index_by_symbol: HashMap<&str, usize> = symbols
            .iter()
            .enumerate()
            .map(|(idx, symbol)| (symbol.as_str(), idx))
            .collect();
        let mut counts = vec![0_i32; symbols.len()];
        for (idx, count) in core_counts.into_iter().enumerate() {
            counts[idx] += count;
        }
        for (symbol, count) in adduct_symbols.iter().zip(adduct_counts.iter()) {
            let Some(idx) = index_by_symbol.get(symbol.as_str()) else {
                return Err(format!(
                    "adduct symbol '{symbol}' missing from isotope symbol list"
                ));
            };
            counts[*idx] += *count;
        }
        if counts.iter().any(|count| *count < 0) {
            return Ok((Vec::new(), Vec::new()));
        }

        let scoring = self.isotope_table.build_scoring_input(
            symbols,
            Vec::new(),
            Vec::new(),
            0.0,
            simulated_mz_tolerance,
            simulated_intensity_threshold,
            charge,
            electron_mass,
        )?;
        isospec_ffi::simulate_isotope_envelope(&counts, &scoring)
    }

    pub fn build_decompose_input(
        &self,
        query_mass: f64,
        charge: i32,
        ppm_error: f64,
        mz_error: f64,
        min_counts: Vec<i64>,
        max_counts: Vec<f64>,
        max_results: i32,
        rdbe_coeffs: Vec<f64>,
        rdbe_min: f64,
        rdbe_max: f64,
        check_octet: bool,
        charge_parity_even: Option<bool>,
        do_rdbe_filter: bool,
        do_iso_filter: bool,
        iso_m1_coeffs: Vec<f64>,
        iso_m2_direct: Vec<f64>,
        obs_m1_ratio: f64,
        obs_m2_ratio: f64,
        iso_tol_rel: f64,
        iso_tol_abs: f64,
        ion_query_mass: f64,
        adduct_mass: f64,
        compute_rdbe: bool,
        electron_mass: f64,
    ) -> Result<DecomposeInput, String> {
        let n_elements = self.n_elements();
        if min_counts.len() != n_elements || max_counts.len() != n_elements {
            return Err("min_counts/max_counts must match element count".to_string());
        }
        if rdbe_coeffs.len() != n_elements
            || iso_m1_coeffs.len() != n_elements
            || iso_m2_direct.len() != n_elements
        {
            return Err("coefficient arrays must match element count".to_string());
        }

        let mut bounds = Vec::with_capacity(n_elements);
        for (max_count, min_count) in max_counts.iter().zip(min_counts.iter()) {
            bounds.push(*max_count - (*min_count as f64));
        }

        let adjusted_mass = query_mass + electron_mass * (charge as f64);
        let (original_min_mass, original_max_mass) =
            self.mass_range(adjusted_mass, ppm_error, mz_error, None);
        let (min_mass, max_mass) =
            self.mass_range(adjusted_mass, ppm_error, mz_error, Some(&min_counts));

        if max_mass > 0.0 {
            for (idx, mass) in self.real_masses.iter().enumerate() {
                let mass_bound = (max_mass / mass).floor();
                bounds[idx] = bounds[idx].min(mass_bound);
            }
        }

        let (min_int, max_int) =
            self.mass_range_as_integers(min_mass.max(0.0), max_mass.max(0.0))?;

        Ok(DecomposeInput {
            ert: Arc::clone(&self.ert),
            integer_masses: Arc::clone(&self.integer_masses),
            real_masses: Arc::clone(&self.real_masses),
            bounds,
            min_values: min_counts,
            min_int,
            max_int,
            original_min_mass,
            original_max_mass,
            charge_mass_offset: electron_mass * (charge as f64),
            max_results,
            rdbe_coeffs,
            rdbe_min,
            rdbe_max,
            check_octet,
            charge_parity_even: charge_parity_even.unwrap_or_else(|| charge.abs() % 2 == 0),
            do_rdbe_filter,
            do_iso_filter,
            iso_m1_coeffs,
            iso_m2_direct,
            obs_m1_ratio,
            obs_m2_ratio,
            iso_tol_rel,
            iso_tol_abs,
            query_mass: ion_query_mass,
            adduct_mass,
            compute_rdbe,
        })
    }

    pub fn find_formulae(
        &self,
        query_mass: f64,
        charge: i32,
        ppm_error: f64,
        mz_error: f64,
        min_counts: Vec<i64>,
        max_counts: Vec<f64>,
        max_results: i32,
        rdbe_coeffs: Vec<f64>,
        rdbe_min: f64,
        rdbe_max: f64,
        check_octet: bool,
        charge_parity_even: Option<bool>,
        do_rdbe_filter: bool,
        do_iso_filter: bool,
        iso_m1_coeffs: Vec<f64>,
        iso_m2_direct: Vec<f64>,
        obs_m1_ratio: f64,
        obs_m2_ratio: f64,
        iso_tol_rel: f64,
        iso_tol_abs: f64,
        ion_query_mass: f64,
        adduct_mass: f64,
        compute_rdbe: bool,
        can_compute_rdbe: bool,
        electron_mass: f64,
        remaining_apply_rdbe_filter: bool,
        remaining_rdbe_min: f64,
        remaining_rdbe_max: f64,
        remaining_check_octet: bool,
        adduct_present: bool,
        adduct_symbols: Vec<String>,
        adduct_counts: Vec<i32>,
        unknown_symbol_indices: Vec<usize>,
        isotope: Option<StoredIsotopeQueryInput>,
    ) -> Result<FindFormulaeOutput, String> {
        let decompose = self.build_decompose_input(
            query_mass,
            charge,
            ppm_error,
            mz_error,
            min_counts,
            max_counts,
            max_results,
            rdbe_coeffs,
            rdbe_min,
            rdbe_max,
            check_octet,
            charge_parity_even,
            do_rdbe_filter,
            do_iso_filter,
            iso_m1_coeffs,
            iso_m2_direct,
            obs_m1_ratio,
            obs_m2_ratio,
            iso_tol_rel,
            iso_tol_abs,
            ion_query_mass,
            adduct_mass,
            compute_rdbe,
            electron_mass,
        )?;

        query::find_formulae(FindFormulaeInput {
            decompose,
            core_symbols: self.element_symbols.clone(),
            charge,
            remaining_apply_rdbe_filter,
            remaining_rdbe_min,
            remaining_rdbe_max,
            remaining_check_octet,
            can_compute_rdbe,
            adduct_present,
            adduct_symbols,
            adduct_counts,
            unknown_symbol_indices,
            isotope: isotope.map(|iso| IsotopeQueryInput {
                symbols: iso.symbols,
                scoring: iso.scoring,
                minimum_rmse: iso.minimum_rmse,
            }),
        })
    }

    fn mass_range(
        &self,
        mass: f64,
        ppm_error: f64,
        mz_error: f64,
        min_counts: Option<&[i64]>,
    ) -> (f64, f64) {
        let ppm_component = mass * ppm_error / 1e6;
        let error = if mz_error > ppm_component {
            mz_error
        } else {
            ppm_component
        };
        let mut min_mass = mass - error;
        let mut max_mass = mass + error;

        if let Some(min_counts) = min_counts {
            for (idx, min_count) in min_counts.iter().enumerate() {
                if *min_count > 0 {
                    let reduce_by = self.real_masses[idx] * (*min_count as f64);
                    min_mass -= reduce_by;
                    max_mass -= reduce_by;
                }
            }
        }

        (min_mass, max_mass)
    }

    fn mass_range_as_integers(&self, min_mass: f64, max_mass: f64) -> Result<(i64, i64), String> {
        let from_int = ((1.0 + self.min_error) * min_mass / self.precision).ceil();
        let to_int = ((1.0 + self.max_error) * max_mass / self.precision).floor();

        if from_int > i64::MAX as f64 || to_int > i64::MAX as f64 {
            return Err(format!(
                "Mass range ({min_mass} - {max_mass}) too large to decompose with current precision: {}",
                self.precision
            ));
        }

        let start = 0_i64.max(from_int as i64);
        let end = start.max(to_int as i64);
        Ok((start, end))
    }
}

pub struct PublicFindFormulaeOutput {
    pub output: FindFormulaeOutput,
    pub adduct_mass: f64,
    pub adduct_symbols: Vec<String>,
    pub adduct_counts: Vec<i32>,
}

fn observed_m1_m2_ratios(
    observed_mz: &[f64],
    observed_intensity: &[f64],
) -> Result<(f64, f64), String> {
    if observed_mz.len() != observed_intensity.len() {
        return Err(
            "observed isotope m/z and intensity arrays must have the same length".to_string(),
        );
    }
    if observed_mz.is_empty() {
        return Ok((0.0, 0.0));
    }

    let mut mono_idx = 0_usize;
    let mut mono_mz = observed_mz[0];
    for (idx, mz) in observed_mz.iter().enumerate().skip(1) {
        if *mz < mono_mz {
            mono_mz = *mz;
            mono_idx = idx;
        }
    }
    let mono_intensity = observed_intensity[mono_idx];
    let mut obs_m1_ratio = 0.0;
    let mut obs_m2_ratio = 0.0;

    for (mz, intensity) in observed_mz.iter().zip(observed_intensity.iter()) {
        let delta = *mz - mono_mz;
        if (0.9..=1.1).contains(&delta) {
            obs_m1_ratio = *intensity / mono_intensity;
        } else if (1.9..=2.1).contains(&delta) {
            obs_m2_ratio = *intensity / mono_intensity;
        }
    }

    Ok((obs_m1_ratio, obs_m2_ratio))
}

pub struct StoredIsotopeQueryInput {
    pub symbols: Vec<String>,
    pub scoring: IsotopeScoringInput,
    pub minimum_rmse: f64,
}

#[cfg(test)]
mod tests {
    use super::StoredFormulaFinder;

    fn finder() -> StoredFormulaFinder {
        StoredFormulaFinder::new(
            vec!["A".to_string(), "B".to_string()],
            vec![vec![0.0, 0.0]],
            vec![1, 2],
            vec![1.0, 2.0],
            1.0,
            0.0,
            0.0,
            vec!["A".to_string(), "B".to_string()],
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            String::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn stored_finder_parses_bounds_and_adducts() {
        let finder = finder();
        assert_eq!(finder.parse_min_counts("A2B*").unwrap(), vec![2, 0]);
        let max = finder.parse_max_counts("A2B*").unwrap();
        assert_eq!(max[0], 2.0);
        assert!(max[1].is_infinite());

        let adduct = finder.parse_adduct("-A2").unwrap();
        assert_eq!(adduct.symbols, vec!["A"]);
        assert_eq!(adduct.counts, vec![-2]);
        assert_eq!(adduct.mass, -2.0);
    }

    #[test]
    fn constructs_from_element_masses_without_python_ert() {
        let finder = StoredFormulaFinder::from_element_masses(
            vec!["B".to_string(), "A".to_string()],
            vec![2.0, 1.0],
            vec!["A".to_string(), "B".to_string()],
            vec![1.0, 2.0],
            vec!["A".to_string(), "B".to_string()],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            String::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        assert_eq!(finder.element_symbols, vec!["A", "B"]);
        assert_eq!(*finder.integer_masses, vec![1, 2]);
        assert_eq!(*finder.real_masses, vec![1.0, 2.0]);
        assert_eq!(*finder.ert, vec![0, 0]);
    }

    #[test]
    fn constructs_setup_data_from_element_isotope_sources() {
        let source_symbols = vec!["A".to_string(), "B".to_string()];
        let source_counts = vec![3, 1];
        let source_mass_numbers = vec![1, 2, 3, 2];
        let source_masses = vec![1.0, 1.1, 1.2, 2.0];
        let source_abundances = vec![100.0, 1.0, 4.0, 50.0];

        let finder = StoredFormulaFinder::from_element_sources(
            vec!["B".to_string(), "A".to_string()],
            source_symbols.clone(),
            source_counts.clone(),
            source_mass_numbers.clone(),
            source_masses.clone(),
            source_abundances.clone(),
            String::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        assert_eq!(finder.element_symbols, vec!["A", "B"]);
        assert_eq!(*finder.real_masses, vec![1.0, 2.0]);
        assert_eq!(finder.element_masses["A"], 1.0);
        assert_eq!(finder.iso_m1_coeffs, vec![0.01, 0.0]);
        assert_eq!(finder.iso_m2_direct, vec![0.04, 0.0]);

        let precomputed = StoredFormulaFinder::from_precomputed_sources(
            vec!["A".to_string(), "B".to_string()],
            vec![vec![0.0, 0.0]],
            vec![1, 2],
            vec![1.0, 2.0],
            1.0,
            0.0,
            0.0,
            source_symbols,
            source_counts,
            source_mass_numbers,
            source_masses,
            source_abundances,
            String::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(precomputed.iso_m1_coeffs, vec![0.01, 0.0]);
        assert_eq!(precomputed.iso_m2_direct, vec![0.04, 0.0]);
    }

    #[test]
    fn constructs_from_embedded_setup_sources() {
        let finder = StoredFormulaFinder::from_embedded_sources(
            vec!["O".to_string(), "H".to_string(), "C".to_string()],
            String::new(),
        )
        .unwrap();

        assert_eq!(finder.element_symbols, vec!["H", "C", "O"]);
        assert_eq!(finder.n_elements(), 3);
        assert!((finder.element_masses["C"] - 12.0).abs() < 1e-12);
        assert!(finder.iso_m1_coeffs[1] > 0.0);
        assert!(finder.isotope_table.isotope_numbers.len() > finder.n_elements());
    }

    #[test]
    fn stored_finder_builds_isotope_query_from_stored_table() {
        let finder = StoredFormulaFinder::new(
            vec!["A".to_string(), "B".to_string()],
            vec![vec![0.0, 0.0]],
            vec![1, 2],
            vec![1.0, 2.0],
            1.0,
            0.0,
            0.0,
            vec!["A".to_string(), "B".to_string(), "X".to_string()],
            vec![1.0, 2.0, 3.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            "/tmp/libisospec.so".to_string(),
            vec!["A".to_string(), "B".to_string(), "X".to_string()],
            vec![2, 1, 2],
            vec![1.0, 1.1, 2.0, 3.0, 3.1],
            vec![0.9, 0.1, 1.0, 0.8, 0.2],
        )
        .unwrap();

        let query = finder
            .build_isotope_query_input(
                &["X".to_string()],
                1,
                0.00054858,
                vec![100.0, 101.0],
                vec![1.0, 0.2],
                0.01,
                0.05,
                0.001,
                0.03,
            )
            .unwrap();

        assert_eq!(query.symbols, vec!["A", "B", "X"]);
        assert_eq!(query.minimum_rmse, 0.03);
        assert_eq!(query.scoring.lib_path, "/tmp/libisospec.so");
        assert_eq!(query.scoring.isotope_numbers, vec![2, 1, 2]);
        assert_eq!(query.scoring.flat_masses, vec![1.0, 1.1, 2.0, 3.0, 3.1]);
        assert_eq!(query.scoring.flat_probs, vec![0.9, 0.1, 1.0, 0.8, 0.2]);
    }

    #[test]
    fn isotope_envelope_simulation_returns_empty_for_negative_ion_counts() {
        let finder = finder();
        let (mz, intensity) = finder
            .simulate_isotope_envelope(
                vec![0, 0],
                vec!["A".to_string()],
                vec![-1],
                0,
                0.00054858,
                0.05,
                0.001,
            )
            .unwrap();

        assert!(mz.is_empty());
        assert!(intensity.is_empty());
    }

    #[test]
    fn builds_decompose_input_with_min_count_mass_shift() {
        let input = finder()
            .build_decompose_input(
                5.0,
                0,
                0.0,
                0.0,
                vec![1, 0],
                vec![10.0, 10.0],
                100,
                vec![0.0, 0.0],
                f64::NEG_INFINITY,
                f64::INFINITY,
                false,
                None,
                false,
                false,
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                0.0,
                0.0,
                0.3,
                0.02,
                5.0,
                0.0,
                false,
                0.0,
            )
            .unwrap();

        assert_eq!(input.min_values, vec![1, 0]);
        assert_eq!(input.min_int, 4);
        assert_eq!(input.max_int, 4);
        assert_eq!(input.bounds, vec![4.0, 2.0]);
    }

    #[test]
    fn mass_range_matches_python_nan_max_semantics() {
        let finder = finder();

        let (min_mass, max_mass) = finder.mass_range(5.0, f64::NAN, 0.1, None);
        assert!(min_mass.is_nan());
        assert!(max_mass.is_nan());
        assert_eq!(
            finder
                .mass_range_as_integers(0.0_f64.max(min_mass), 0.0_f64.max(max_mass))
                .unwrap(),
            (0, 0)
        );

        let (min_mass, max_mass) = finder.mass_range(f64::INFINITY, 0.0, 0.1, None);
        assert!(min_mass.is_nan());
        assert!(max_mass.is_nan());
        assert_eq!(
            finder
                .mass_range_as_integers(0.0_f64.max(min_mass), 0.0_f64.max(max_mass))
                .unwrap(),
            (0, 0)
        );
    }

    #[test]
    fn stored_finder_runs_full_query_pipeline() {
        let output = finder()
            .find_formulae(
                2.0,
                0,
                0.0,
                0.0,
                vec![0, 0],
                vec![10.0, 10.0],
                10,
                vec![0.0, 0.0],
                f64::NEG_INFINITY,
                f64::INFINITY,
                false,
                None,
                false,
                false,
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                0.0,
                0.0,
                0.3,
                0.02,
                2.0,
                0.0,
                false,
                false,
                0.0,
                false,
                0.0,
                0.0,
                false,
                false,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap();

        assert_eq!(output.counts.to_rows(), vec![vec![2, 0], vec![0, 1]]);
    }

    #[test]
    fn public_query_owns_count_vectors_and_adduct_adjustment() {
        let output = finder()
            .find_formulae_public(
                3.0,
                0,
                0.0,
                0.0,
                None,
                vec!["A".to_string()],
                vec![0.0],
                Some("A0B1".to_string()),
                Vec::new(),
                Vec::new(),
                10,
                None,
                false,
                Some("A".to_string()),
                false,
                Vec::new(),
                Vec::new(),
                0.3,
                0.02,
                0.0,
                false,
                Vec::new(),
                Vec::new(),
                0.0,
                0.0,
                0.0,
                0.0,
            )
            .unwrap();

        assert_eq!(output.output.counts.to_rows(), vec![vec![0, 1]]);
        assert_eq!(output.adduct_mass, 1.0);
        assert_eq!(output.adduct_symbols, vec!["A"]);
        assert_eq!(output.adduct_counts, vec![1]);
    }

    #[test]
    fn configured_query_owns_rdbe_prefilter_inputs() {
        let finder = StoredFormulaFinder::new(
            vec!["H".to_string(), "C".to_string()],
            vec![vec![0.0, 0.0]],
            vec![1, 12],
            vec![1.0, 12.0],
            1.0,
            0.0,
            0.0,
            vec!["H".to_string(), "C".to_string()],
            vec![1.0, 12.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            String::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();

        let output = finder
            .find_formulae_configured(
                78.0,
                0,
                0.0,
                0.0,
                vec![0, 0],
                vec![100.0, 100.0],
                100,
                true,
                4.0,
                4.0,
                true,
                false,
                Vec::new(),
                Vec::new(),
                0.3,
                0.02,
                78.0,
                0.0,
                0.0,
                false,
                Vec::new(),
                Vec::new(),
                false,
                Vec::new(),
                Vec::new(),
                0.0,
                0.0,
                0.0,
                0.0,
            )
            .unwrap();

        assert_eq!(output.counts.to_rows(), vec![vec![6, 6]]);
        assert_eq!(output.rdbe, None);
        assert_eq!(
            crate::chemistry::rdbe_from_counts_i32(output.counts.row(0), &output.rdbe_coeffs),
            4.0
        );
    }
}
