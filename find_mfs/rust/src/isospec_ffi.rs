use std::ffi::c_void;
use std::os::raw::c_int;

use libloading::Library;
use rayon::prelude::*;

type SetupIso =
    unsafe extern "C" fn(c_int, *const i32, *const i32, *const f64, *const f64) -> *mut c_void;
type SetupThresholdFixedEnvelope =
    unsafe extern "C" fn(*mut c_void, f64, c_int, c_int) -> *mut c_void;
type ConfsNoFixedEnvelope = unsafe extern "C" fn(*mut c_void) -> usize;
type GetArrayFixedEnvelope = unsafe extern "C" fn(*mut c_void) -> *const f64;
type DeleteFixedEnvelope = unsafe extern "C" fn(*mut c_void, c_int);
type DeleteIso = unsafe extern "C" fn(*mut c_void);
type FreeReleasedArray = unsafe extern "C" fn(*mut c_void);

struct IsoSpecLibrary {
    _lib: Library,
    setup_iso: SetupIso,
    setup_threshold: SetupThresholdFixedEnvelope,
    confs_no: ConfsNoFixedEnvelope,
    get_masses: GetArrayFixedEnvelope,
    get_probs: GetArrayFixedEnvelope,
    delete_fe: DeleteFixedEnvelope,
    delete_iso: DeleteIso,
    free_array: FreeReleasedArray,
}

impl IsoSpecLibrary {
    fn load(path: &str) -> Result<Self, String> {
        let lib = unsafe { Library::new(path) }
            .map_err(|err| format!("Cannot load IsoSpec library at {path}: {err}"))?;

        unsafe {
            let setup_iso = *lib
                .get::<SetupIso>(b"setupIso\0")
                .map_err(|err| format!("Failed to load setupIso: {err}"))?;
            let setup_threshold = *lib
                .get::<SetupThresholdFixedEnvelope>(b"setupThresholdFixedEnvelope\0")
                .map_err(|err| format!("Failed to load setupThresholdFixedEnvelope: {err}"))?;
            let confs_no = *lib
                .get::<ConfsNoFixedEnvelope>(b"confs_noFixedEnvelope\0")
                .map_err(|err| format!("Failed to load confs_noFixedEnvelope: {err}"))?;
            let get_masses = *lib
                .get::<GetArrayFixedEnvelope>(b"massesFixedEnvelope\0")
                .map_err(|err| format!("Failed to load massesFixedEnvelope: {err}"))?;
            let get_probs = *lib
                .get::<GetArrayFixedEnvelope>(b"probsFixedEnvelope\0")
                .map_err(|err| format!("Failed to load probsFixedEnvelope: {err}"))?;
            let delete_fe = *lib
                .get::<DeleteFixedEnvelope>(b"deleteFixedEnvelope\0")
                .map_err(|err| format!("Failed to load deleteFixedEnvelope: {err}"))?;
            let delete_iso = *lib
                .get::<DeleteIso>(b"deleteIso\0")
                .map_err(|err| format!("Failed to load deleteIso: {err}"))?;
            let free_array = *lib
                .get::<FreeReleasedArray>(b"freeReleasedArray\0")
                .map_err(|err| format!("Failed to load freeReleasedArray: {err}"))?;

            Ok(Self {
                _lib: lib,
                setup_iso,
                setup_threshold,
                confs_no,
                get_masses,
                get_probs,
                delete_fe,
                delete_iso,
                free_array,
            })
        }
    }
}

pub struct IsotopeScoringInput {
    pub lib_path: String,
    pub isotope_numbers: Vec<i32>,
    pub flat_masses: Vec<f64>,
    pub flat_probs: Vec<f64>,
    pub observed_mz: Vec<f64>,
    pub observed_intensity: Vec<f64>,
    pub mz_match_tolerance: f64,
    pub simulated_mz_tolerance: f64,
    pub simulated_intensity_threshold: f64,
    pub charge: i32,
    pub electron_mass: f64,
}

pub struct IsotopeScoreOutput {
    pub rmse: Vec<f64>,
    pub match_fraction: Vec<f64>,
    pub n_matched: Vec<i32>,
    pub peak_matches: Vec<Vec<i8>>,
}

const RAYON_ISOTOPE_MIN_CANDIDATES: usize = 32;

pub fn score_isotope_batch(
    counts_2d: &[Vec<i32>],
    input: &IsotopeScoringInput,
) -> Result<IsotopeScoreOutput, String> {
    validate_scoring_input(counts_2d, input)?;

    let lib = IsoSpecLibrary::load(&input.lib_path)?;
    let n_obs = input.observed_mz.len();
    let iso_offsets = isotope_offsets(&input.isotope_numbers);

    let rows: Vec<(f64, f64, i32, Vec<i8>)> =
        if counts_2d.len() >= RAYON_ISOTOPE_MIN_CANDIDATES {
            counts_2d
                .par_iter()
                .map(|counts| {
                    let mut matches = vec![0_i8; n_obs];
                    let (r, mf, nm) =
                        score_candidate_zeroskip(&lib, counts, &iso_offsets, input, &mut matches)?;
                    Ok((r, mf, nm, matches))
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            let mut rows = Vec::with_capacity(counts_2d.len());
            for counts in counts_2d {
                let mut matches = vec![0_i8; n_obs];
                let (r, mf, nm) =
                    score_candidate_zeroskip(&lib, counts, &iso_offsets, input, &mut matches)?;
                rows.push((r, mf, nm, matches));
            }
            rows
        };

    let mut rmse = Vec::with_capacity(rows.len());
    let mut match_fraction = Vec::with_capacity(rows.len());
    let mut n_matched = Vec::with_capacity(rows.len());
    let mut peak_matches = Vec::with_capacity(rows.len());

    for (r, mf, nm, matches) in rows {
        rmse.push(r);
        match_fraction.push(mf);
        n_matched.push(nm);
        peak_matches.push(matches);
    }

    Ok(IsotopeScoreOutput {
        rmse,
        match_fraction,
        n_matched,
        peak_matches,
    })
}

pub fn simulate_isotope_envelope(
    candidate_counts: &[i32],
    input: &IsotopeScoringInput,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    validate_simulation_input(candidate_counts, input)?;

    let lib = IsoSpecLibrary::load(&input.lib_path)?;
    let iso_offsets = isotope_offsets(&input.isotope_numbers);
    simulate_candidate_zeroskip(&lib, candidate_counts, &iso_offsets, input)
}

fn validate_scoring_input(
    counts_2d: &[Vec<i32>],
    input: &IsotopeScoringInput,
) -> Result<(), String> {
    let n_elements = input.isotope_numbers.len();
    if input.observed_mz.len() != input.observed_intensity.len() {
        return Err("observed m/z and intensity arrays must have the same length".to_string());
    }
    if input.observed_mz.is_empty() {
        return Err("observed envelope must contain at least one peak".to_string());
    }
    if input.flat_masses.len() != input.flat_probs.len() {
        return Err("flat isotope masses and probabilities must have the same length".to_string());
    }
    let expected_isotopes: usize = input.isotope_numbers.iter().map(|n| *n as usize).sum();
    if expected_isotopes != input.flat_masses.len() {
        return Err(format!(
            "flat isotope arrays have length {}, expected {}",
            input.flat_masses.len(),
            expected_isotopes
        ));
    }
    for (idx, counts) in counts_2d.iter().enumerate() {
        if counts.len() != n_elements {
            return Err(format!(
                "candidate {idx} has {} element counts, expected {n_elements}",
                counts.len()
            ));
        }
    }
    Ok(())
}

fn validate_simulation_input(
    candidate_counts: &[i32],
    input: &IsotopeScoringInput,
) -> Result<(), String> {
    let n_elements = input.isotope_numbers.len();
    if candidate_counts.len() != n_elements {
        return Err(format!(
            "candidate has {} element counts, expected {n_elements}",
            candidate_counts.len()
        ));
    }
    if input.flat_masses.len() != input.flat_probs.len() {
        return Err("flat isotope masses and probabilities must have the same length".to_string());
    }
    let expected_isotopes: usize = input.isotope_numbers.iter().map(|n| *n as usize).sum();
    if expected_isotopes != input.flat_masses.len() {
        return Err(format!(
            "flat isotope arrays have length {}, expected {}",
            input.flat_masses.len(),
            expected_isotopes
        ));
    }
    Ok(())
}

fn isotope_offsets(isotope_numbers: &[i32]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(isotope_numbers.len() + 1);
    offsets.push(0);
    for n_iso in isotope_numbers {
        let next = offsets.last().copied().unwrap_or(0) + (*n_iso as usize);
        offsets.push(next);
    }
    offsets
}

fn score_candidate_zeroskip(
    lib: &IsoSpecLibrary,
    candidate_counts: &[i32],
    isotope_offsets: &[usize],
    input: &IsotopeScoringInput,
    peak_matches_out: &mut [i8],
) -> Result<(f64, f64, i32), String> {
    peak_matches_out.fill(0);

    let (combined_mz, combined_int) =
        simulate_candidate_zeroskip(lib, candidate_counts, isotope_offsets, input)?;
    if combined_mz.is_empty() {
        return Ok((1.0, 0.0, 0));
    }
    Ok(score_observed_peaks(
        &input.observed_mz,
        &input.observed_intensity,
        &combined_mz,
        &combined_int,
        input.mz_match_tolerance,
        peak_matches_out,
    ))
}

fn simulate_candidate_zeroskip(
    lib: &IsoSpecLibrary,
    candidate_counts: &[i32],
    isotope_offsets: &[usize],
    input: &IsotopeScoringInput,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut active_iso_numbers = Vec::new();
    let mut active_counts = Vec::new();
    let mut active_masses = Vec::new();
    let mut active_probs = Vec::new();

    for (idx, count) in candidate_counts.iter().enumerate() {
        if *count > 0 {
            active_iso_numbers.push(input.isotope_numbers[idx]);
            active_counts.push(*count);
            let start = isotope_offsets[idx];
            let end = isotope_offsets[idx + 1];
            active_masses.extend_from_slice(&input.flat_masses[start..end]);
            active_probs.extend_from_slice(&input.flat_probs[start..end]);
        }
    }

    if active_counts.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    simulate_single_envelope(
        lib,
        &active_iso_numbers,
        &active_counts,
        &active_masses,
        &active_probs,
        input,
    )
}

fn simulate_single_envelope(
    lib: &IsoSpecLibrary,
    iso_numbers: &[i32],
    atom_counts: &[i32],
    flat_masses: &[f64],
    flat_probs: &[f64],
    input: &IsotopeScoringInput,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    unsafe {
        let iso_ptr = (lib.setup_iso)(
            iso_numbers.len() as c_int,
            iso_numbers.as_ptr(),
            atom_counts.as_ptr(),
            flat_masses.as_ptr(),
            flat_probs.as_ptr(),
        );
        if iso_ptr.is_null() {
            return Err("IsoSpec setupIso returned null".to_string());
        }

        let env_ptr = (lib.setup_threshold)(iso_ptr, input.simulated_intensity_threshold, 0, 0);
        if env_ptr.is_null() {
            (lib.delete_iso)(iso_ptr);
            return Err("IsoSpec setupThresholdFixedEnvelope returned null".to_string());
        }

        let n_peaks = (lib.confs_no)(env_ptr);
        if n_peaks == 0 {
            (lib.delete_fe)(env_ptr, 0);
            (lib.delete_iso)(iso_ptr);
            return Ok((Vec::new(), Vec::new()));
        }

        let masses_raw = (lib.get_masses)(env_ptr);
        let probs_raw = (lib.get_probs)(env_ptr);
        if masses_raw.is_null() || probs_raw.is_null() {
            if !masses_raw.is_null() {
                (lib.free_array)(masses_raw as *mut c_void);
            }
            if !probs_raw.is_null() {
                (lib.free_array)(probs_raw as *mut c_void);
            }
            (lib.delete_fe)(env_ptr, 0);
            (lib.delete_iso)(iso_ptr);
            return Err("IsoSpec returned null mass/probability arrays".to_string());
        }

        let mut pred_mz = std::slice::from_raw_parts(masses_raw, n_peaks).to_vec();
        let mut pred_prob = std::slice::from_raw_parts(probs_raw, n_peaks).to_vec();

        (lib.free_array)(masses_raw as *mut c_void);
        (lib.free_array)(probs_raw as *mut c_void);
        (lib.delete_fe)(env_ptr, 0);
        (lib.delete_iso)(iso_ptr);

        if input.charge != 0 {
            let abs_charge = input.charge.abs() as f64;
            let charge_offset = (input.charge as f64) * input.electron_mass;
            for mz in &mut pred_mz {
                *mz = (*mz - charge_offset) / abs_charge;
            }
        }

        sort_parallel_by_mz(&mut pred_mz, &mut pred_prob);
        Ok(combine_unresolved(
            &pred_mz,
            &pred_prob,
            input.simulated_mz_tolerance,
        ))
    }
}

fn sort_parallel_by_mz(mz: &mut [f64], prob: &mut [f64]) {
    let mut pairs: Vec<(f64, f64)> = mz.iter().copied().zip(prob.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Greater));
    for (idx, (mass, probability)) in pairs.into_iter().enumerate() {
        mz[idx] = mass;
        prob[idx] = probability;
    }
}

fn combine_unresolved(mz: &[f64], prob: &[f64], tolerance: f64) -> (Vec<f64>, Vec<f64>) {
    let mut combined_mz = Vec::new();
    let mut combined_int = Vec::new();
    let mut i = 0;

    while i < mz.len() {
        let mut mz_sum = mz[i] * prob[i];
        let mut int_sum = prob[i];
        let mut j = i + 1;
        while j < mz.len() && (mz[j] - mz[i]).abs() <= tolerance {
            mz_sum += mz[j] * prob[j];
            int_sum += prob[j];
            j += 1;
        }
        combined_mz.push(mz_sum / int_sum);
        combined_int.push(int_sum);
        i = j;
    }

    if let Some(max_intensity) = combined_int.iter().copied().reduce(f64::max) {
        if max_intensity > 0.0 {
            for intensity in &mut combined_int {
                *intensity /= max_intensity;
            }
        }
    }

    (combined_mz, combined_int)
}

fn score_observed_peaks(
    obs_mz: &[f64],
    obs_int: &[f64],
    pred_mz: &[f64],
    pred_int: &[f64],
    match_tolerance: f64,
    peak_matches_out: &mut [i8],
) -> (f64, f64, i32) {
    let mut base_idx = 0_usize;
    let mut max_obs = obs_int[0];
    for (idx, intensity) in obs_int.iter().enumerate().skip(1) {
        if *intensity > max_obs {
            max_obs = *intensity;
            base_idx = idx;
        }
    }

    let mut sse = 0.0;
    let mut count = 0_i32;
    let mut n_matched = 0_i32;

    for (i, observed_mz) in obs_mz.iter().enumerate() {
        let mut best_diff = f64::INFINITY;
        let mut best_j = 0_usize;
        for (j, predicted_mz) in pred_mz.iter().enumerate() {
            let diff = (*observed_mz - *predicted_mz).abs();
            if diff < best_diff {
                best_diff = diff;
                best_j = j;
            }
        }

        let mut pred_val = 0.0;
        let matched = best_diff <= match_tolerance;
        if matched {
            pred_val = pred_int[best_j];
            n_matched += 1;
        }

        peak_matches_out[i] = if matched { 1 } else { 0 };

        if i != base_idx {
            let delta = obs_int[i] - pred_val;
            sse += delta * delta;
            count += 1;
        }
    }

    let rmse = if count > 0 {
        (sse / (count as f64)).sqrt()
    } else {
        0.0
    };
    let match_fraction = if obs_mz.is_empty() {
        0.0
    } else {
        (n_matched as f64) / (obs_mz.len() as f64)
    };

    (rmse, match_fraction, n_matched)
}

#[cfg(test)]
mod tests {
    use super::{combine_unresolved, score_observed_peaks};

    #[test]
    fn combine_unresolved_matches_weighted_average_and_rescale() {
        let (mz, intensity) = combine_unresolved(&[10.0, 10.01, 11.0], &[0.25, 0.75, 0.5], 0.05);
        assert_eq!(mz.len(), 2);
        assert!((mz[0] - 10.0075).abs() < 1e-12);
        assert_eq!(intensity, vec![1.0, 0.5]);
    }

    #[test]
    fn scoring_excludes_observed_base_peak_from_rmse() {
        let mut matches = vec![0_i8; 2];
        let (rmse, mf, nm) = score_observed_peaks(
            &[100.0, 101.0],
            &[1.0, 0.25],
            &[100.0, 101.0],
            &[1.0, 0.10],
            0.01,
            &mut matches,
        );
        assert!((rmse - 0.15).abs() < 1e-12);
        assert_eq!(mf, 1.0);
        assert_eq!(nm, 2);
        assert_eq!(matches, vec![1, 1]);
    }
}
