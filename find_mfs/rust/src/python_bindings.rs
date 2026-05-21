use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::chemistry;
use crate::finder::StoredFormulaFinder;
use crate::formula;
use crate::fragmentation_spectrum::{
    compute_sirius_like_tree_from_spectrum, SiriusLikeConfig, SiriusLikeTables, SpectrumPeak,
    SpectrumTreeResult,
};
use crate::fragmentation_tree::{
    compute_fragmentation_tree, FragmentCandidate, GraphScoring, SubFormulaGraphInput,
    TreeSolveOptions, TreeSolver,
};
use crate::prior::PriorScorer;
use crate::query::FindFormulaeOutput;

type CountInputTuple = (Option<String>, Vec<String>, Vec<f64>);
type AdductElementTuple = (Vec<String>, Vec<i32>);
type QueryRowTuple = (
    Vec<i32>,
    f64,
    f64,
    f64,
    Option<f64>,
    Option<(f64, f64, i32, Vec<i8>)>,
    String,
);
type DisplayRowTuple = (
    String,
    f64,
    f64,
    Option<f64>,
    Option<String>,
    Option<f64>,
    Option<f64>,
);
type PublicResultTuple = (PyRustQueryResult, (f64, Vec<String>, Vec<i32>));
type PyFragmentCandidateTuple = (String, Vec<i32>, String, usize, i32, f64, f64);
type PySelectedLossTuple = (String, String, f64);
type PySelectedRawFragmentTuple = (
    String,
    Vec<i32>,
    String,
    Option<usize>,
    i32,
    f64,
    f64,
    Option<f64>,
);
type PySelectedRawLossTuple = (String, String, f64);
type PyFragmentationTreeTuple = (
    f64,
    bool,
    String,
    String,
    Vec<String>,
    Vec<PySelectedLossTuple>,
    usize,
    usize,
    usize,
    usize,
);
type PyRawSpectrumTreeTuple = (
    f64,
    bool,
    String,
    String,
    Vec<PySelectedRawFragmentTuple>,
    Vec<PySelectedRawLossTuple>,
    usize,
    usize,
    usize,
    usize,
    f64,
);

struct PyIsotopeMatchInput {
    enable_iso_prefilter: bool,
    observed_mz_for_prefilter: Vec<f64>,
    observed_intensity_for_prefilter: Vec<f64>,
    iso_tol_rel: f64,
    iso_tol_abs: f64,
    do_isotope_match: bool,
    observed_mz: Vec<f64>,
    observed_intensity: Vec<f64>,
    mz_match_tolerance: f64,
    simulated_mz_tolerance: f64,
    simulated_intensity_threshold: f64,
    minimum_rmse: f64,
}

#[pyclass(name = "RustQueryResult")]
#[derive(Clone)]
struct PyRustQueryResult {
    output: FindFormulaeOutput,
    prior_score: Option<Vec<f64>>,
    posterior_score: Option<Vec<f64>>,
}

impl PyRustQueryResult {
    fn new(output: FindFormulaeOutput) -> Self {
        Self {
            output,
            prior_score: None,
            posterior_score: None,
        }
    }

    fn len_internal(&self) -> usize {
        self.output.counts.len()
    }

    fn from_indices(&self, indices: &[usize]) -> PyResult<Self> {
        let len = self.len_internal();
        for index in indices {
            if *index >= len {
                return Err(PyValueError::new_err(format!(
                    "result index {index} out of range for {len} candidates"
                )));
            }
        }

        Ok(Self {
            output: FindFormulaeOutput {
                core_symbols: self.output.core_symbols.clone(),
                formula_charge: self.output.formula_charge,
                rdbe_coeffs: self.output.rdbe_coeffs.clone(),
                counts: self.output.counts.take_rows(indices),
                exact_masses: take_vec(&self.output.exact_masses, indices),
                error_ppm: take_vec(&self.output.error_ppm, indices),
                error_da: take_vec(&self.output.error_da, indices),
                rdbe: self
                    .output
                    .rdbe
                    .as_ref()
                    .map(|values| take_vec(values, indices)),
                iso_rmse: self
                    .output
                    .iso_rmse
                    .as_ref()
                    .map(|values| take_vec(values, indices)),
                iso_match_frac: self
                    .output
                    .iso_match_frac
                    .as_ref()
                    .map(|values| take_vec(values, indices)),
                iso_n_matched: self
                    .output
                    .iso_n_matched
                    .as_ref()
                    .map(|values| take_vec(values, indices)),
                iso_peak_matches: self
                    .output
                    .iso_peak_matches
                    .as_ref()
                    .map(|values| take_vec(values, indices)),
                formula_strings: self
                    .output
                    .formula_strings
                    .as_ref()
                    .map(|values| take_vec(values, indices)),
            },
            prior_score: self
                .prior_score
                .as_ref()
                .map(|values| take_vec(values, indices)),
            posterior_score: self
                .posterior_score
                .as_ref()
                .map(|values| take_vec(values, indices)),
        })
    }

    fn from_mask(&self, mask: &[bool]) -> PyResult<Self> {
        if mask.len() != self.len_internal() {
            return Err(PyValueError::new_err(format!(
                "mask has length {}, expected {}",
                mask.len(),
                self.len_internal()
            )));
        }
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(idx, keep)| if *keep { Some(idx) } else { None })
            .collect();
        self.from_indices(&indices)
    }

    fn empty_like(&self) -> PyResult<Self> {
        self.from_indices(&[])
    }

    fn display_limit(&self, max_rows: Option<isize>) -> usize {
        let len = self.len_internal();
        match max_rows {
            None => len,
            Some(value) if value >= 0 => (value as usize).min(len),
            Some(value) => len.saturating_sub(value.unsigned_abs()),
        }
    }

    fn display_row(&self, idx: usize, n_observed: usize) -> DisplayRowTuple {
        (
            self.formula_string(idx),
            self.output.error_ppm[idx],
            self.output.error_da[idx],
            self.rdbe_value(idx),
            self.output
                .iso_n_matched
                .as_ref()
                .map(|values| format!("{}/{}", values[idx], n_observed)),
            self.output.iso_rmse.as_ref().map(|values| values[idx]),
            self.prior_score.as_ref().map(|values| values[idx]),
        )
    }

    fn resolve_index(&self, idx: isize) -> PyResult<usize> {
        let len = self.len_internal();
        let idx = if idx < 0 {
            len.checked_sub(idx.unsigned_abs())
        } else {
            Some(idx as usize)
        };
        let Some(idx) = idx else {
            return Err(PyValueError::new_err("result index out of range"));
        };
        if idx >= len {
            return Err(PyValueError::new_err(format!(
                "result index {idx} out of range for {len} candidates"
            )));
        }
        Ok(idx)
    }

    fn passes_python_octet_rule(rdbe: f64, charge: i32) -> bool {
        let is_half_integer = (2.0 * rdbe).rem_euclid(2.0) == 1.0;
        if charge.abs() % 2 == 0 {
            !is_half_integer
        } else {
            is_half_integer
        }
    }

    fn formula_string(&self, idx: usize) -> String {
        if let Some(formula_strings) = self.output.formula_strings.as_ref() {
            return formula_strings[idx].clone();
        }
        let counts: Vec<i64> = self
            .output
            .counts
            .row(idx)
            .iter()
            .map(|count| *count as i64)
            .collect();
        formula::format_formula_from_counts(
            &self.output.core_symbols,
            &counts,
            self.output.formula_charge,
        )
    }

    fn rdbe_value(&self, idx: usize) -> Option<f64> {
        if let Some(rdbe) = self.output.rdbe.as_ref() {
            return Some(rdbe[idx]);
        }
        if self.output.rdbe_coeffs.is_empty() {
            return None;
        }
        Some(chemistry::rdbe_from_counts_i32(
            self.output.counts.row(idx),
            &self.output.rdbe_coeffs,
        ))
    }
}

#[pyclass(name = "RustFragmentationTreeResult")]
#[derive(Clone)]
struct PyRustFragmentationTreeResult {
    result: SpectrumTreeResult,
}

impl PyRustFragmentationTreeResult {
    fn new(result: SpectrumTreeResult) -> Self {
        Self { result }
    }

    fn fragment_rows(&self) -> Vec<PySelectedRawFragmentTuple> {
        self.result
            .fragments
            .iter()
            .map(|fragment| {
                (
                    fragment.formula.clone(),
                    fragment.counts.clone(),
                    fragment.ionization.clone(),
                    fragment.peak_id,
                    fragment.color,
                    fragment.mass,
                    fragment.score,
                    fragment.intensity,
                )
            })
            .collect()
    }

    fn loss_rows(&self) -> Vec<PySelectedRawLossTuple> {
        self.result
            .losses
            .iter()
            .map(|loss| {
                (
                    loss.source_formula.clone(),
                    loss.target_formula.clone(),
                    loss.score,
                )
            })
            .collect()
    }

    fn tuple(&self) -> PyRawSpectrumTreeTuple {
        (
            self.result.tree_score,
            self.result.is_optimal,
            self.result.solver_status.clone(),
            self.result.root_formula.clone(),
            self.fragment_rows(),
            self.loss_rows(),
            self.result.graph_vertex_count,
            self.result.graph_edge_count,
            self.result.reduced_vertex_count,
            self.result.reduced_edge_count,
            self.result.tree_size_score,
        )
    }
}

#[pymethods]
impl PyRustFragmentationTreeResult {
    #[getter]
    fn tree_score(&self) -> f64 {
        self.result.tree_score
    }

    #[getter]
    fn is_optimal(&self) -> bool {
        self.result.is_optimal
    }

    #[getter]
    fn solver_status(&self) -> String {
        self.result.solver_status.clone()
    }

    #[getter]
    fn root_formula(&self) -> String {
        self.result.root_formula.clone()
    }

    #[getter]
    fn graph_vertex_count(&self) -> usize {
        self.result.graph_vertex_count
    }

    #[getter]
    fn graph_edge_count(&self) -> usize {
        self.result.graph_edge_count
    }

    #[getter]
    fn reduced_vertex_count(&self) -> usize {
        self.result.reduced_vertex_count
    }

    #[getter]
    fn reduced_edge_count(&self) -> usize {
        self.result.reduced_edge_count
    }

    #[getter]
    fn tree_size_score(&self) -> f64 {
        self.result.tree_size_score
    }

    fn fragments(&self) -> Vec<PySelectedRawFragmentTuple> {
        self.fragment_rows()
    }

    fn losses(&self) -> Vec<PySelectedRawLossTuple> {
        self.loss_rows()
    }

    fn formula_strings(&self) -> Vec<String> {
        self.result
            .fragments
            .iter()
            .map(|fragment| fragment.formula.clone())
            .collect()
    }

    fn loss_tuples(&self) -> Vec<(String, String)> {
        self.result
            .losses
            .iter()
            .map(|loss| (loss.source_formula.clone(), loss.target_formula.clone()))
            .collect()
    }

    fn graph_counts(&self) -> (usize, usize, usize, usize) {
        (
            self.result.graph_vertex_count,
            self.result.graph_edge_count,
            self.result.reduced_vertex_count,
            self.result.reduced_edge_count,
        )
    }

    fn to_tuple(&self) -> PyRawSpectrumTreeTuple {
        self.tuple()
    }

    fn __repr__(&self) -> String {
        format!(
            "RustFragmentationTreeResult(root={:?}, fragments={}, losses={}, tree_score={:.6}, status={:?})",
            self.result.root_formula,
            self.result.fragments.len(),
            self.result.losses.len(),
            self.result.tree_score,
            self.result.solver_status,
        )
    }
}

fn take_vec<T: Clone>(values: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|index| values[*index].clone()).collect()
}

#[pymethods]
impl PyRustQueryResult {
    fn __len__(&self) -> usize {
        self.len_internal()
    }

    fn n_observed(&self) -> usize {
        self.output
            .iso_peak_matches
            .as_ref()
            .and_then(|rows| rows.first())
            .map(|row| row.len())
            .unwrap_or(0)
    }

    fn formula_strings(&self) -> Vec<String> {
        if let Some(formula_strings) = self.output.formula_strings.as_ref() {
            return formula_strings.clone();
        }
        (0..self.len_internal())
            .map(|idx| self.formula_string(idx))
            .collect()
    }

    #[pyo3(signature = (max_rows=None))]
    fn table_rows(&self, max_rows: Option<isize>) -> Vec<DisplayRowTuple> {
        let n_observed = self.n_observed();
        let limit = self.display_limit(max_rows);
        (0..limit)
            .map(|idx| self.display_row(idx, n_observed))
            .collect()
    }

    fn row(&self, idx: isize) -> PyResult<QueryRowTuple> {
        let idx = self.resolve_index(idx)?;

        let isotope = self.output.iso_rmse.as_ref().map(|rmse| {
            (
                rmse[idx],
                self.output
                    .iso_match_frac
                    .as_ref()
                    .map(|values| values[idx])
                    .unwrap_or(0.0),
                self.output
                    .iso_n_matched
                    .as_ref()
                    .map(|values| values[idx])
                    .unwrap_or(0),
                self.output
                    .iso_peak_matches
                    .as_ref()
                    .map(|values| values[idx].clone())
                    .unwrap_or_default(),
            )
        });

        Ok((
            self.output.counts.row(idx).to_vec(),
            self.output.exact_masses[idx],
            self.output.error_ppm[idx],
            self.output.error_da[idx],
            self.rdbe_value(idx),
            isotope,
            self.formula_string(idx),
        ))
    }

    fn take_indices(&self, indices: Vec<usize>) -> PyResult<Self> {
        self.from_indices(&indices)
    }

    fn score_values(&self, idx: isize) -> PyResult<(Option<f64>, Option<f64>)> {
        let idx = self.resolve_index(idx)?;
        Ok((
            self.prior_score.as_ref().map(|values| values[idx]),
            self.posterior_score.as_ref().map(|values| values[idx]),
        ))
    }

    fn sort_by_error(&self, reverse: bool) -> PyResult<Self> {
        let mut indices: Vec<usize> = (0..self.len_internal()).collect();
        indices.sort_by(|left, right| {
            self.output.error_da[*left]
                .abs()
                .partial_cmp(&self.output.error_da[*right].abs())
                .unwrap_or(Ordering::Equal)
        });
        if reverse {
            indices.reverse();
        }
        self.from_indices(&indices)
    }

    fn sort_by_rmse(&self, reverse: bool) -> PyResult<Self> {
        let Some(rmse) = self.output.iso_rmse.as_ref() else {
            return Ok(self.clone());
        };
        let mut indices: Vec<usize> = (0..self.len_internal()).collect();
        indices.sort_by(|left, right| {
            rmse[*left]
                .partial_cmp(&rmse[*right])
                .unwrap_or(Ordering::Equal)
        });
        if reverse {
            indices.reverse();
        }
        self.from_indices(&indices)
    }

    fn sort_by_prior(&self, reverse: bool) -> PyResult<Self> {
        let Some(scores) = self.prior_score.as_ref() else {
            return Ok(self.clone());
        };
        let mut indices: Vec<usize> = (0..self.len_internal()).collect();
        if reverse {
            indices.sort_by(|left, right| {
                scores[*left]
                    .partial_cmp(&scores[*right])
                    .unwrap_or(Ordering::Equal)
            });
        } else {
            indices.sort_by(|left, right| {
                scores[*right]
                    .partial_cmp(&scores[*left])
                    .unwrap_or(Ordering::Equal)
            });
        }
        self.from_indices(&indices)
    }

    fn sort_by_posterior(&self, reverse: bool) -> PyResult<Self> {
        let Some(scores) = self.posterior_score.as_ref() else {
            return Ok(self.clone());
        };
        let mut indices: Vec<usize> = (0..self.len_internal()).collect();
        if reverse {
            indices.sort_by(|left, right| {
                scores[*left]
                    .partial_cmp(&scores[*right])
                    .unwrap_or(Ordering::Equal)
            });
        } else {
            indices.sort_by(|left, right| {
                scores[*right]
                    .partial_cmp(&scores[*left])
                    .unwrap_or(Ordering::Equal)
            });
        }
        self.from_indices(&indices)
    }

    #[allow(clippy::too_many_arguments)]
    fn score_prior(
        &self,
        core_symbols: Vec<String>,
        ratio_elements: Vec<String>,
        p_absent: Vec<f64>,
        kde_points: Vec<Vec<f64>>,
        kde_weights: Vec<Vec<f64>>,
        kde_variance: Vec<f64>,
        uniform_weight: f64,
        mass_sigma_ppm: f64,
        isotope_sigma: f64,
    ) -> PyResult<Self> {
        if !self.output.counts.is_empty() && core_symbols.len() != self.output.counts.n_cols() {
            return Err(PyValueError::new_err(
                "core symbol count does not match result count vectors",
            ));
        }
        let prior_scorer = PriorScorer::new(
            &core_symbols,
            &ratio_elements,
            &p_absent,
            &kde_points,
            &kde_weights,
            &kde_variance,
            uniform_weight,
        )
        .map_err(PyValueError::new_err)?;

        let mut prior_scores = Vec::with_capacity(self.len_internal());
        let mut posterior_scores = Vec::with_capacity(self.len_internal());
        for (idx, counts) in self.output.counts.rows().enumerate() {
            let prior_score = prior_scorer.score_counts(counts);
            let mut posterior_score = prior_score;
            posterior_score -= self.output.error_ppm[idx].powi(2) / (2.0 * mass_sigma_ppm.powi(2));
            if let Some(rmse) = self.output.iso_rmse.as_ref() {
                posterior_score -= rmse[idx].powi(2) / (2.0 * isotope_sigma.powi(2));
            }
            prior_scores.push(prior_score);
            posterior_scores.push(posterior_score);
        }

        Ok(Self {
            output: self.output.clone(),
            prior_score: Some(prior_scores),
            posterior_score: Some(posterior_scores),
        })
    }

    fn filter_by_rdbe(&self, min_rdbe: f64, max_rdbe: f64) -> PyResult<Self> {
        let mask: Vec<bool> = (0..self.len_internal())
            .map(|idx| {
                self.rdbe_value(idx)
                    .map(|value| value >= min_rdbe && value <= max_rdbe)
                    .unwrap_or(false)
            })
            .collect();
        self.from_mask(&mask)
    }

    #[pyo3(signature = (max_ppm=None, max_da=None))]
    fn filter_by_error(&self, max_ppm: Option<f64>, max_da: Option<f64>) -> PyResult<Self> {
        let mask: Vec<bool> = (0..self.len_internal())
            .map(|idx| {
                let ppm_ok = max_ppm
                    .map(|limit| self.output.error_ppm[idx].abs() <= limit)
                    .unwrap_or(true);
                let da_ok = max_da
                    .map(|limit| self.output.error_da[idx].abs() <= limit)
                    .unwrap_or(true);
                ppm_ok && da_ok
            })
            .collect();
        self.from_mask(&mask)
    }

    fn filter_by_isotope_quality(
        &self,
        max_match_rmse: f64,
        min_match_fraction: f64,
    ) -> PyResult<Self> {
        let (Some(rmse), Some(match_fraction)) = (
            self.output.iso_rmse.as_ref(),
            self.output.iso_match_frac.as_ref(),
        ) else {
            return self.empty_like();
        };
        let mask: Vec<bool> = rmse
            .iter()
            .zip(match_fraction.iter())
            .map(|(rmse, fraction)| *rmse <= max_match_rmse && *fraction >= min_match_fraction)
            .collect();
        self.from_mask(&mask)
    }

    fn filter_by_octet(&self, charge: i32) -> PyResult<Self> {
        let mask: Vec<bool> = (0..self.len_internal())
            .map(|idx| {
                self.rdbe_value(idx)
                    .map(|value| Self::passes_python_octet_rule(value, charge))
                    .unwrap_or(false)
            })
            .collect();
        self.from_mask(&mask)
    }
}

fn extract_count_input(value: &Bound<'_, PyAny>) -> PyResult<CountInputTuple> {
    if value.is_none() {
        return Ok((None, Vec::new(), Vec::new()));
    }
    if let Ok(formula) = value.extract::<String>() {
        return Ok((Some(formula), Vec::new(), Vec::new()));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut symbols = Vec::with_capacity(dict.len());
        let mut values = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            symbols.push(key.extract::<String>()?);
            values.push(value.extract::<f64>()?);
        }
        return Ok((None, symbols, values));
    }
    Err(PyValueError::new_err(
        "count bounds must be None, a formula string, or a dict",
    ))
}

fn extract_adduct_elements(value: &Bound<'_, PyAny>) -> PyResult<AdductElementTuple> {
    if value.is_none() {
        return Ok((Vec::new(), Vec::new()));
    }
    if let Ok((symbols, counts)) = value.extract::<AdductElementTuple>() {
        if symbols.len() != counts.len() {
            return Err(PyValueError::new_err(
                "adduct symbol/count arrays must match",
            ));
        }
        let mut filtered_symbols = Vec::with_capacity(symbols.len());
        let mut filtered_counts = Vec::with_capacity(counts.len());
        for (symbol, count) in symbols.into_iter().zip(counts.into_iter()) {
            if count != 0 {
                filtered_symbols.push(symbol);
                filtered_counts.push(count);
            }
        }
        return Ok((filtered_symbols, filtered_counts));
    }
    let dict = value.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err("adduct elements must be None, a dict, or (symbols, counts)")
    })?;
    let mut symbols = Vec::with_capacity(dict.len());
    let mut counts = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let count = value.extract::<i32>()?;
        if count != 0 {
            symbols.push(key.extract::<String>()?);
            counts.push(count);
        }
    }
    Ok((symbols, counts))
}

fn extract_optional_float_attr(value: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let attr = value.getattr(name)?;
    if attr.is_none() {
        Ok(0.0)
    } else {
        attr.extract::<f64>()
    }
}

fn extract_envelope(value: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let rows = match value.extract::<Vec<Vec<f64>>>() {
        Ok(rows) => rows,
        Err(_) => value.call_method0("tolist")?.extract::<Vec<Vec<f64>>>()?,
    };
    let mut mz = Vec::with_capacity(rows.len());
    let mut intensity = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != 2 {
            return Err(PyValueError::new_err(
                "isotope envelope rows must contain [m/z, intensity]",
            ));
        }
        mz.push(row[0]);
        intensity.push(row[1]);
    }
    Ok((mz, intensity))
}

fn extract_isotope_match_input(
    value: &Bound<'_, PyAny>,
    mass: f64,
) -> PyResult<PyIsotopeMatchInput> {
    if value.is_none() {
        return Ok(PyIsotopeMatchInput {
            enable_iso_prefilter: false,
            observed_mz_for_prefilter: Vec::new(),
            observed_intensity_for_prefilter: Vec::new(),
            iso_tol_rel: 0.3,
            iso_tol_abs: 0.02,
            do_isotope_match: false,
            observed_mz: Vec::new(),
            observed_intensity: Vec::new(),
            mz_match_tolerance: 0.0,
            simulated_mz_tolerance: 0.0,
            simulated_intensity_threshold: 0.0,
            minimum_rmse: 0.0,
        });
    }

    let envelope = value.getattr("envelope")?;
    let (observed_mz, observed_intensity) = extract_envelope(&envelope)?;
    let enable_iso_prefilter = value
        .getattr("enable_approx_prefilter")?
        .extract::<bool>()?;
    let (observed_mz_for_prefilter, observed_intensity_for_prefilter) = if enable_iso_prefilter {
        (observed_mz.clone(), observed_intensity.clone())
    } else {
        (Vec::new(), Vec::new())
    };
    let ppm_to_da = 1e-6 * extract_optional_float_attr(value, "mz_tolerance_ppm")? * mass;
    let mz_match_tolerance = extract_optional_float_attr(value, "mz_tolerance_da")?.max(ppm_to_da);

    Ok(PyIsotopeMatchInput {
        enable_iso_prefilter,
        observed_mz_for_prefilter,
        observed_intensity_for_prefilter,
        iso_tol_rel: value.getattr("approx_tolerance_rel")?.extract::<f64>()?,
        iso_tol_abs: value.getattr("approx_tolerance_abs")?.extract::<f64>()?,
        do_isotope_match: true,
        observed_mz,
        observed_intensity,
        mz_match_tolerance,
        simulated_mz_tolerance: value.getattr("simulated_mz_tolerance")?.extract::<f64>()?,
        simulated_intensity_threshold: value
            .getattr("simulated_intensity_threshold")?
            .extract::<f64>()?,
        minimum_rmse: value.getattr("minimum_rmse")?.extract::<f64>()?,
    })
}

#[pyclass(name = "RustFormulaFinder")]
struct PyRustFormulaFinder {
    inner: Arc<StoredFormulaFinder>,
    sirius_like_tables: Option<Arc<SiriusLikeTables>>,
}

#[pymethods]
impl PyRustFormulaFinder {
    #[new]
    fn new(
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
    ) -> PyResult<Self> {
        let inner = StoredFormulaFinder::new(
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
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
            sirius_like_tables: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_element_masses(
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
    ) -> PyResult<Self> {
        let inner = StoredFormulaFinder::from_element_masses(
            element_symbols,
            element_masses_for_finder,
            element_mass_symbols,
            element_mass_values,
            isotope_coeff_symbols,
            isotope_m1_values,
            isotope_m2_values,
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
            sirius_like_tables: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_precomputed_sources(
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
    ) -> PyResult<Self> {
        let inner = StoredFormulaFinder::from_precomputed_sources(
            element_symbols,
            ert,
            integer_masses,
            real_masses,
            precision,
            min_error,
            max_error,
            element_source_symbols,
            element_source_isotope_numbers,
            flat_element_mass_numbers,
            flat_element_isotope_masses,
            flat_element_isotope_abundances,
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
            sirius_like_tables: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_element_sources(
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
    ) -> PyResult<Self> {
        let inner = StoredFormulaFinder::from_element_sources(
            element_symbols,
            element_source_symbols,
            element_source_isotope_numbers,
            flat_element_mass_numbers,
            flat_element_isotope_masses,
            flat_element_isotope_abundances,
            isospec_lib_path,
            isotope_table_symbols,
            isotope_numbers,
            flat_isotope_masses,
            flat_isotope_probs,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
            sirius_like_tables: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_precomputed_embedded_sources(
        element_symbols: Vec<String>,
        ert: Vec<Vec<f64>>,
        integer_masses: Vec<i64>,
        real_masses: Vec<f64>,
        precision: f64,
        min_error: f64,
        max_error: f64,
        isospec_lib_path: String,
    ) -> PyResult<Self> {
        let inner = StoredFormulaFinder::from_precomputed_embedded_sources(
            element_symbols,
            ert,
            integer_masses,
            real_masses,
            precision,
            min_error,
            max_error,
            isospec_lib_path,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
            sirius_like_tables: None,
        })
    }

    #[staticmethod]
    fn from_embedded_sources(
        element_symbols: Vec<String>,
        isospec_lib_path: String,
    ) -> PyResult<Self> {
        let inner = StoredFormulaFinder::from_embedded_sources(element_symbols, isospec_lib_path)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Arc::new(inner),
            sirius_like_tables: None,
        })
    }

    fn with_sirius_like_tables(&self, tables: Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::clone(&self.inner),
            sirius_like_tables: Some(Arc::new(extract_sirius_like_tables(&tables)?)),
        })
    }

    fn has_custom_sirius_like_tables(&self) -> bool {
        self.sirius_like_tables.is_some()
    }

    fn element_symbols(&self) -> Vec<String> {
        self.inner.element_symbols.clone()
    }

    fn simulate_isotope_envelope(
        &self,
        py: Python<'_>,
        core_counts: Vec<i32>,
        adduct_symbols: Vec<String>,
        adduct_counts: Vec<i32>,
        charge: i32,
        electron_mass: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            inner.simulate_isotope_envelope(
                core_counts,
                adduct_symbols,
                adduct_counts,
                charge,
                electron_mass,
                simulated_mz_tolerance,
                simulated_intensity_threshold,
            )
        })
        .map_err(PyValueError::new_err)
    }

    fn simulate_isotope_envelope_python(
        &self,
        py: Python<'_>,
        core_counts: Vec<i32>,
        adduct_elements: Bound<'_, PyAny>,
        charge: i32,
        electron_mass: f64,
        simulated_mz_tolerance: f64,
        simulated_intensity_threshold: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (adduct_symbols, adduct_counts) = extract_adduct_elements(&adduct_elements)?;
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            inner.simulate_isotope_envelope(
                core_counts,
                adduct_symbols,
                adduct_counts,
                charge,
                electron_mass,
                simulated_mz_tolerance,
                simulated_intensity_threshold,
            )
        })
        .map_err(PyValueError::new_err)
    }

    fn parse_min_counts(&self, formula: &str) -> PyResult<Vec<i64>> {
        self.inner
            .parse_min_counts(formula)
            .map_err(PyValueError::new_err)
    }

    fn parse_max_counts(&self, formula: &str) -> PyResult<Vec<f64>> {
        self.inner
            .parse_max_counts(formula)
            .map_err(PyValueError::new_err)
    }

    fn parse_adduct(&self, adduct: &str) -> PyResult<(f64, Vec<String>, Vec<i32>)> {
        let parsed = self
            .inner
            .parse_adduct(adduct)
            .map_err(PyValueError::new_err)?;
        Ok((parsed.mass, parsed.symbols, parsed.counts))
    }

    #[allow(clippy::too_many_arguments)]
    fn find_formulae_public_result_python(
        &self,
        py: Python<'_>,
        mass: f64,
        charge: i32,
        ppm_error: f64,
        mz_error: f64,
        min_counts: Bound<'_, PyAny>,
        max_counts: Bound<'_, PyAny>,
        max_results: i32,
        apply_rdbe_filter: bool,
        rdbe_min: f64,
        rdbe_max: f64,
        check_octet: bool,
        adduct: String,
        isotope_match: Bound<'_, PyAny>,
        electron_mass: f64,
    ) -> PyResult<PublicResultTuple> {
        let (min_count_formula, min_count_symbols, min_count_values) =
            extract_count_input(&min_counts)?;
        let (max_count_formula, max_count_symbols, max_count_values) =
            extract_count_input(&max_counts)?;
        let isotope = extract_isotope_match_input(&isotope_match, mass)?;
        let finder = Arc::clone(&self.inner);

        let output = py
            .allow_threads(move || {
                finder.find_formulae_public(
                    mass,
                    charge,
                    ppm_error,
                    mz_error,
                    min_count_formula,
                    min_count_symbols,
                    min_count_values,
                    max_count_formula,
                    max_count_symbols,
                    max_count_values,
                    max_results,
                    if apply_rdbe_filter {
                        Some((rdbe_min, rdbe_max))
                    } else {
                        None
                    },
                    check_octet,
                    if adduct.is_empty() {
                        None
                    } else {
                        Some(adduct)
                    },
                    isotope.enable_iso_prefilter,
                    isotope.observed_mz_for_prefilter,
                    isotope.observed_intensity_for_prefilter,
                    isotope.iso_tol_rel,
                    isotope.iso_tol_abs,
                    electron_mass,
                    isotope.do_isotope_match,
                    isotope.observed_mz,
                    isotope.observed_intensity,
                    isotope.mz_match_tolerance,
                    isotope.simulated_mz_tolerance,
                    isotope.simulated_intensity_threshold,
                    isotope.minimum_rmse,
                )
            })
            .map_err(PyValueError::new_err)?;

        Ok((
            PyRustQueryResult::new(output.output),
            (
                output.adduct_mass,
                output.adduct_symbols,
                output.adduct_counts,
            ),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        precursor_mz,
        precursor_formula,
        precursor_ion,
        peaks,
        config,
        reduce_graph,
        minimal_score=None,
        time_limit_seconds=None,
        threads=None,
        electron_mass=0.000548579909065,
        solver="highs".to_string()
    ))]
    fn find_fragmentation_tree_from_spectrum_python(
        &self,
        py: Python<'_>,
        precursor_mz: f64,
        precursor_formula: String,
        precursor_ion: String,
        peaks: Vec<(f64, f64)>,
        config: Bound<'_, PyAny>,
        reduce_graph: bool,
        minimal_score: Option<f64>,
        time_limit_seconds: Option<f64>,
        threads: Option<u32>,
        electron_mass: f64,
        solver: String,
    ) -> PyResult<PyRawSpectrumTreeTuple> {
        let result = self.find_fragmentation_tree_from_spectrum_result_python(
            py,
            precursor_mz,
            precursor_formula,
            precursor_ion,
            peaks,
            config,
            reduce_graph,
            minimal_score,
            time_limit_seconds,
            threads,
            electron_mass,
            solver,
        )?;
        Ok(result.tuple())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        precursor_mz,
        precursor_formula,
        precursor_ion,
        peaks,
        config,
        reduce_graph,
        minimal_score=None,
        time_limit_seconds=None,
        threads=None,
        electron_mass=0.000548579909065,
        solver="highs".to_string()
    ))]
    fn find_fragmentation_tree_from_spectrum_result_python(
        &self,
        py: Python<'_>,
        precursor_mz: f64,
        precursor_formula: String,
        precursor_ion: String,
        peaks: Vec<(f64, f64)>,
        config: Bound<'_, PyAny>,
        reduce_graph: bool,
        minimal_score: Option<f64>,
        time_limit_seconds: Option<f64>,
        threads: Option<u32>,
        electron_mass: f64,
        solver: String,
    ) -> PyResult<PyRustFragmentationTreeResult> {
        let config = extract_sirius_like_config(&config)?;
        let solver = TreeSolver::from_name(&solver).map_err(PyValueError::new_err)?;
        let peaks = peaks
            .into_iter()
            .map(|(mz, intensity)| SpectrumPeak { mz, intensity })
            .collect();
        let options = TreeSolveOptions {
            minimal_score,
            time_limit_seconds,
            threads,
            solver,
        };
        let finder = Arc::clone(&self.inner);
        let tables = self.sirius_like_tables.clone();

        let result = py
            .allow_threads(move || {
                compute_sirius_like_tree_from_spectrum(
                    &finder,
                    precursor_mz,
                    &precursor_formula,
                    &precursor_ion,
                    peaks,
                    config,
                    tables,
                    options,
                    reduce_graph,
                    electron_mass,
                )
            })
            .map_err(PyValueError::new_err)?;

        Ok(PyRustFragmentationTreeResult::new(result))
    }
}

fn extract_sirius_like_config(config: &Bound<'_, PyAny>) -> PyResult<SiriusLikeConfig> {
    Ok(SiriusLikeConfig {
        ms2_tolerance_ppm: config.getattr("ms2_tolerance_ppm")?.extract()?,
        candidate_search_ppm: config.getattr("candidate_search_ppm")?.extract()?,
        candidate_search_absolute_da: config.getattr("candidate_search_absolute_da")?.extract()?,
        precursor_tolerance_ppm: config.getattr("precursor_tolerance_ppm")?.extract()?,
        candidate_limit_per_peak: config.getattr("candidate_limit_per_peak")?.extract()?,
        max_fragment_peaks: config.getattr("max_fragment_peaks")?.extract()?,
        min_relative_intensity: config.getattr("min_relative_intensity")?.extract()?,
        merge_close_peaks: config.getattr("merge_close_peaks")?.extract()?,
        median_noise_intensity: config.getattr("median_noise_intensity")?.extract()?,
        tree_size_score: config.getattr("tree_size_score")?.extract()?,
        fragment_size_max_score: config.getattr("fragment_size_max_score")?.extract()?,
        fragment_size_max_mz: config.getattr("fragment_size_max_mz")?.extract()?,
        clipped_noise_xmin: config.getattr("clipped_noise_xmin")?.extract()?,
        clipped_noise_beta: config.getattr("clipped_noise_beta")?.extract()?,
        loss_size_mean: config.getattr("loss_size_mean")?.extract()?,
        loss_size_variance: config.getattr("loss_size_variance")?.extract()?,
        loss_size_normalization: config.getattr("loss_size_normalization")?.extract()?,
        intrinsically_charged_root_penalty: config
            .getattr("intrinsically_charged_root_penalty")?
            .extract()?,
        strange_element_root_penalty: config.getattr("strange_element_root_penalty")?.extract()?,
        strange_element_small_fragment_score: config
            .getattr("strange_element_small_fragment_score")?
            .extract()?,
        strange_element_small_fragment_max_mass: config
            .getattr("strange_element_small_fragment_max_mass")?
            .extract()?,
        strange_element_fragment_score: config
            .getattr("strange_element_fragment_score")?
            .extract()?,
        strange_element_fragment_penalty: config
            .getattr("strange_element_fragment_penalty")?
            .extract()?,
        strange_element_fragment_min_mass: config
            .getattr("strange_element_fragment_min_mass")?
            .extract()?,
        strange_element_loss_score: config.getattr("strange_element_loss_score")?.extract()?,
        free_radical_penalty: config.getattr("free_radical_penalty")?.extract()?,
        free_radical_normalization: config.getattr("free_radical_normalization")?.extract()?,
        strict_sirius_radical_parity: config.getattr("strict_sirius_radical_parity")?.extract()?,
        dbe_loss_score: config.getattr("dbe_loss_score")?.extract()?,
        pure_carbon_nitrogen_loss_penalty: config
            .getattr("pure_carbon_nitrogen_loss_penalty")?
            .extract()?,
        mass_deviation_vertex_weight: config.getattr("mass_deviation_vertex_weight")?.extract()?,
        mass_deviation_edge_weight: config.getattr("mass_deviation_edge_weight")?.extract()?,
        mass_deviation_absolute_da: config.getattr("mass_deviation_absolute_da")?.extract()?,
        loss_mass_deviation_absolute_da: config
            .getattr("loss_mass_deviation_absolute_da")?
            .extract()?,
        chemical_prior_root_score: config.getattr("chemical_prior_root_score")?.extract()?,
        db_paired_formula_score: config.getattr("db_paired_formula_score")?.extract()?,
        db_paired_formulas: extract_optional_string_set(&config.getattr("db_paired_formulas")?)?,
        enable_common_fragment_score: config.getattr("enable_common_fragment_score")?.extract()?,
        carbohydrogen_root_score: config.getattr("carbohydrogen_root_score")?.extract()?,
        enable_carbohydrogen_fragment_score: config
            .getattr("enable_carbohydrogen_fragment_score")?
            .extract()?,
        carbohydrogen_fragment_min_relative_intensity: config
            .getattr("carbohydrogen_fragment_min_relative_intensity")?
            .extract()?,
        carbohydrogen_fragment_xmin: config.getattr("carbohydrogen_fragment_xmin")?.extract()?,
        carbohydrogen_fragment_median: config
            .getattr("carbohydrogen_fragment_median")?
            .extract()?,
        multimere_root_loss_score: config.getattr("multimere_root_loss_score")?.extract()?,
        multimere_loss_score: config.getattr("multimere_loss_score")?.extract()?,
        fatty_acid_chain_score_weight: config
            .getattr("fatty_acid_chain_score_weight")?
            .extract()?,
        fatty_acid_chain_double_bond_decay: config
            .getattr("fatty_acid_chain_double_bond_decay")?
            .extract()?,
        fatty_acid_chain_min_length: config.getattr("fatty_acid_chain_min_length")?.extract()?,
        fatty_acid_chain_max_length: config.getattr("fatty_acid_chain_max_length")?.extract()?,
        fatty_acid_chain_max_double_bonds: config
            .getattr("fatty_acid_chain_max_double_bonds")?
            .extract()?,
        recombine_common_losses: config.getattr("recombine_common_losses")?.extract()?,
        estimate_tree_size: config.getattr("estimate_tree_size")?.extract()?,
        tree_size_increase: config.getattr("tree_size_increase")?.extract()?,
        max_tree_size_increase: config.getattr("max_tree_size_increase")?.extract()?,
        max_tree_size_score: config.getattr("max_tree_size_score")?.extract()?,
        min_explained_intensity: config.getattr("min_explained_intensity")?.extract()?,
        min_explained_peaks: config.getattr("min_explained_peaks")?.extract()?,
        use_sirius_tree_size_quality_threshold: config
            .getattr("use_sirius_tree_size_quality_threshold")?
            .extract()?,
    })
}

fn extract_optional_string_set(value: &Bound<'_, PyAny>) -> PyResult<Option<HashSet<String>>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<Vec<String>>()?.into_iter().collect()))
}

fn extract_sirius_like_tables(tables: &Bound<'_, PyDict>) -> PyResult<SiriusLikeTables> {
    Ok(SiriusLikeTables {
        common_fragments: extract_string_f64_map(tables, "common_fragments")?,
        common_losses: extract_string_f64_map(tables, "common_losses")?,
        recombined_common_losses: extract_string_f64_map(tables, "recombined_common_losses")?,
        recombined_common_loss_overrides: extract_string_f64_map(
            tables,
            "recombined_common_loss_overrides",
        )?,
        common_radicals: extract_string_f64_map(tables, "common_radicals")?,
        common_root_losses: extract_string_f64_map(tables, "common_root_losses")?,
        strange_fragment_whitelist: extract_string_set(tables, "strange_fragment_whitelist")?,
        strange_losses: extract_string_set(tables, "strange_losses")?,
        common_fragment_normalization: extract_f64(tables, "common_fragment_normalization")?,
        common_loss_normalization: extract_f64(tables, "common_loss_normalization")?,
        common_root_loss_normalization: extract_f64(tables, "common_root_loss_normalization")?,
    })
}

fn extract_string_f64_map(tables: &Bound<'_, PyDict>, key: &str) -> PyResult<HashMap<String, f64>> {
    let Some(value) = tables.get_item(key)? else {
        return Err(PyValueError::new_err(format!("missing table key {key:?}")));
    };
    Ok(value.extract::<Vec<(String, f64)>>()?.into_iter().collect())
}

fn extract_string_set(tables: &Bound<'_, PyDict>, key: &str) -> PyResult<HashSet<String>> {
    let Some(value) = tables.get_item(key)? else {
        return Err(PyValueError::new_err(format!("missing table key {key:?}")));
    };
    Ok(value.extract::<Vec<String>>()?.into_iter().collect())
}

fn extract_f64(tables: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    let Some(value) = tables.get_item(key)? else {
        return Err(PyValueError::new_err(format!("missing table key {key:?}")));
    };
    value.extract()
}

#[pyfunction]
fn format_formula(symbols: Vec<String>, counts: Vec<i64>, charge: i32) -> PyResult<String> {
    if symbols.len() != counts.len() {
        return Err(PyValueError::new_err(
            "symbols and counts must have the same length",
        ));
    }
    Ok(formula::format_formula_from_counts(
        &symbols, &counts, charge,
    ))
}

#[pyfunction]
fn parse_formula_counts(formula_str: &str, symbols: Vec<String>) -> PyResult<Vec<i64>> {
    formula::parse_formula_counts(formula_str, &symbols).map_err(PyValueError::new_err)
}

#[pyfunction]
fn parse_element_symbols(formula_str: &str) -> PyResult<Vec<String>> {
    formula::parse_element_symbols(formula_str).map_err(PyValueError::new_err)
}

fn py_fragment_candidates(values: Vec<PyFragmentCandidateTuple>) -> Vec<FragmentCandidate> {
    values
        .into_iter()
        .map(
            |(formula, counts, ionization, peak_id, color, mass, score)| FragmentCandidate {
                formula,
                counts,
                ionization,
                peak_id,
                color,
                mass,
                score,
            },
        )
        .collect()
}

fn py_color_scores(values: Option<Vec<(i32, f64)>>) -> BTreeMap<i32, f64> {
    values.unwrap_or_default().into_iter().collect()
}

fn py_peak_pair_scores(values: Option<Vec<(i32, i32, f64)>>) -> BTreeMap<(i32, i32), f64> {
    values
        .unwrap_or_default()
        .into_iter()
        .map(|(parent_color, child_color, score)| ((parent_color, child_color), score))
        .collect()
}

fn py_formula_scores(values: Option<Vec<(String, f64)>>) -> BTreeMap<String, f64> {
    values.unwrap_or_default().into_iter().collect()
}

fn py_loss_scores(values: Option<Vec<(String, String, f64)>>) -> BTreeMap<(String, String), f64> {
    values
        .unwrap_or_default()
        .into_iter()
        .map(|(parent_formula, child_formula, score)| ((parent_formula, child_formula), score))
        .collect()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    root_candidates,
    fragment_candidates,
    allowed_ionizations=None,
    peak_scores=None,
    peak_pair_scores=None,
    fragment_scores=None,
    loss_scores=None,
    general_graph_score=0.0,
    reduce_graph=true,
    minimal_score=None,
    time_limit_seconds=None,
    threads=None,
    solver="highs".to_string()
))]
fn solve_fragmentation_tree_python(
    py: Python<'_>,
    root_candidates: Vec<PyFragmentCandidateTuple>,
    fragment_candidates: Vec<PyFragmentCandidateTuple>,
    allowed_ionizations: Option<Vec<String>>,
    peak_scores: Option<Vec<(i32, f64)>>,
    peak_pair_scores: Option<Vec<(i32, i32, f64)>>,
    fragment_scores: Option<Vec<(String, f64)>>,
    loss_scores: Option<Vec<(String, String, f64)>>,
    general_graph_score: f64,
    reduce_graph: bool,
    minimal_score: Option<f64>,
    time_limit_seconds: Option<f64>,
    threads: Option<u32>,
    solver: String,
) -> PyResult<PyFragmentationTreeTuple> {
    let input = SubFormulaGraphInput {
        root_candidates: py_fragment_candidates(root_candidates),
        fragment_candidates: py_fragment_candidates(fragment_candidates),
        allowed_ionizations: allowed_ionizations.unwrap_or_default(),
    };
    let scoring = GraphScoring {
        peak_scores: py_color_scores(peak_scores),
        peak_pair_scores: py_peak_pair_scores(peak_pair_scores),
        fragment_scores: py_formula_scores(fragment_scores),
        loss_scores: py_loss_scores(loss_scores),
        general_graph_score,
    };
    let options = TreeSolveOptions {
        minimal_score,
        time_limit_seconds,
        threads,
        solver: TreeSolver::from_name(&solver).map_err(PyValueError::new_err)?,
    };

    let result = py
        .allow_threads(move || compute_fragmentation_tree(input, &scoring, options, reduce_graph))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let tree_graph = result.reduced_graph.as_ref().unwrap_or(&result.graph);
    let selected_formulas = result
        .tree
        .selected_fragments
        .iter()
        .map(|fragment_id| tree_graph.fragments[*fragment_id].formula.clone())
        .collect();
    let selected_losses = result
        .tree
        .selected_edges
        .iter()
        .map(|edge_id| {
            let edge = &tree_graph.edges[*edge_id];
            (
                tree_graph.fragments[edge.source].formula.clone(),
                tree_graph.fragments[edge.target].formula.clone(),
                edge.weight,
            )
        })
        .collect();
    let root_formula = tree_graph.fragments[result.tree.root_fragment]
        .formula
        .clone();
    let (reduced_vertices, reduced_edges) = result
        .reduced_graph
        .as_ref()
        .map(|graph| (graph.fragments.len(), graph.edges.len()))
        .unwrap_or((0, 0));

    Ok((
        result.tree.tree_weight,
        result.tree.is_optimal,
        format!("{:?}", result.tree.status),
        root_formula,
        selected_formulas,
        selected_losses,
        result.graph.fragments.len(),
        result.graph.edges.len(),
        reduced_vertices,
        reduced_edges,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRustFormulaFinder>()?;
    m.add_class::<PyRustQueryResult>()?;
    m.add_class::<PyRustFragmentationTreeResult>()?;
    m.add_function(wrap_pyfunction!(format_formula, m)?)?;
    m.add_function(wrap_pyfunction!(parse_formula_counts, m)?)?;
    m.add_function(wrap_pyfunction!(parse_element_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(solve_fragmentation_tree_python, m)?)?;
    Ok(())
}
