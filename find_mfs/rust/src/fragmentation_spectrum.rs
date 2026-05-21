use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::chemistry;
use crate::finder::StoredFormulaFinder;
use crate::formula;
use crate::fragmentation_tables::default_sirius_like_tables;
use crate::fragmentation_tree::{
    compute_fragmentation_tree, FragmentCandidate, GraphScoring, SubFormulaGraphInput,
    TreeSolveOptions,
};

#[link(name = "m")]
extern "C" {
    #[link_name = "erfc"]
    fn c_erfc(x: f64) -> f64;
    #[link_name = "log"]
    fn c_log(x: f64) -> f64;
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumPeak {
    pub mz: f64,
    pub intensity: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct ProcessedPeak {
    mz: f64,
    intensity: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct GeneratedCandidate {
    candidate: FragmentCandidate,
    theoretical_mz: f64,
    intensity: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SiriusLikeConfig {
    pub ms2_tolerance_ppm: f64,
    pub candidate_search_ppm: f64,
    pub candidate_search_absolute_da: f64,
    pub precursor_tolerance_ppm: f64,
    pub candidate_limit_per_peak: i32,
    pub max_fragment_peaks: usize,
    pub min_relative_intensity: f64,
    pub merge_close_peaks: bool,
    pub median_noise_intensity: f64,
    pub tree_size_score: f64,
    pub fragment_size_max_score: f64,
    pub fragment_size_max_mz: f64,
    pub clipped_noise_xmin: f64,
    pub clipped_noise_beta: f64,
    pub loss_size_mean: f64,
    pub loss_size_variance: f64,
    pub loss_size_normalization: f64,
    pub intrinsically_charged_root_penalty: f64,
    pub strange_element_root_penalty: f64,
    pub strange_element_small_fragment_score: f64,
    pub strange_element_small_fragment_max_mass: f64,
    pub strange_element_fragment_score: f64,
    pub strange_element_fragment_penalty: f64,
    pub strange_element_fragment_min_mass: f64,
    pub strange_element_loss_score: f64,
    pub free_radical_penalty: f64,
    pub free_radical_normalization: f64,
    pub strict_sirius_radical_parity: bool,
    pub dbe_loss_score: f64,
    pub pure_carbon_nitrogen_loss_penalty: f64,
    pub mass_deviation_vertex_weight: f64,
    pub mass_deviation_edge_weight: f64,
    pub mass_deviation_absolute_da: f64,
    pub loss_mass_deviation_absolute_da: f64,
    pub chemical_prior_root_score: f64,
    pub db_paired_formula_score: f64,
    pub db_paired_formulas: Option<HashSet<String>>,
    pub enable_common_fragment_score: bool,
    pub carbohydrogen_root_score: f64,
    pub enable_carbohydrogen_fragment_score: bool,
    pub carbohydrogen_fragment_min_relative_intensity: f64,
    pub carbohydrogen_fragment_xmin: f64,
    pub carbohydrogen_fragment_median: f64,
    pub multimere_root_loss_score: f64,
    pub multimere_loss_score: f64,
    pub fatty_acid_chain_score_weight: f64,
    pub fatty_acid_chain_double_bond_decay: f64,
    pub fatty_acid_chain_min_length: i32,
    pub fatty_acid_chain_max_length: i32,
    pub fatty_acid_chain_max_double_bonds: i32,
    pub recombine_common_losses: bool,
    pub estimate_tree_size: bool,
    pub tree_size_increase: f64,
    pub max_tree_size_increase: f64,
    pub max_tree_size_score: f64,
    pub min_explained_intensity: f64,
    pub min_explained_peaks: usize,
    pub use_sirius_tree_size_quality_threshold: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiriusLikeTables {
    pub common_fragments: HashMap<String, f64>,
    pub common_losses: HashMap<String, f64>,
    pub recombined_common_losses: HashMap<String, f64>,
    pub recombined_common_loss_overrides: HashMap<String, f64>,
    pub common_radicals: HashMap<String, f64>,
    pub common_root_losses: HashMap<String, f64>,
    pub strange_fragment_whitelist: HashSet<String>,
    pub strange_losses: HashSet<String>,
    pub common_fragment_normalization: f64,
    pub common_loss_normalization: f64,
    pub common_root_loss_normalization: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedRawFragment {
    pub formula: String,
    pub counts: Vec<i32>,
    pub ionization: String,
    pub peak_id: Option<usize>,
    pub color: i32,
    pub mass: f64,
    pub score: f64,
    pub intensity: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedRawLoss {
    pub source_formula: String,
    pub target_formula: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumTreeResult {
    pub tree_score: f64,
    pub is_optimal: bool,
    pub solver_status: String,
    pub root_formula: String,
    pub fragments: Vec<SelectedRawFragment>,
    pub losses: Vec<SelectedRawLoss>,
    pub graph_vertex_count: usize,
    pub graph_edge_count: usize,
    pub reduced_vertex_count: usize,
    pub reduced_edge_count: usize,
    pub tree_size_score: f64,
}

#[derive(Clone, Debug)]
struct Scorer<'a> {
    symbols: &'a [String],
    config: &'a SiriusLikeConfig,
    tables: &'a SiriusLikeTables,
    element_masses: &'a HashMap<String, f64>,
}

pub fn compute_sirius_like_tree_from_spectrum(
    finder: &StoredFormulaFinder,
    precursor_mz: f64,
    precursor_formula: &str,
    precursor_ion: &str,
    peaks: Vec<SpectrumPeak>,
    config: SiriusLikeConfig,
    tables: Option<Arc<SiriusLikeTables>>,
    solve_options: TreeSolveOptions,
    reduce_graph: bool,
    electron_mass: f64,
) -> Result<SpectrumTreeResult, String> {
    if precursor_ion != "[M+H]+" {
        return Err("raw spectrum fragmentation trees currently support only [M+H]+".to_string());
    }
    let root_counts_i64 =
        formula::parse_formula_counts(precursor_formula, &finder.element_symbols)?;
    let root_counts = counts_i64_to_i32(&root_counts_i64)?;
    let tables = tables.as_deref().unwrap_or(default_sirius_like_tables());
    let scorer = Scorer {
        symbols: &finder.element_symbols,
        config: &config,
        tables,
        element_masses: &finder.element_masses,
    };

    if config.estimate_tree_size {
        let mut tree_size = config.tree_size_score;
        let mut increase = 0.0;
        let mut last_result = None;

        while increase <= config.max_tree_size_increase {
            let mut current_config = config.clone();
            current_config.tree_size_score = tree_size;
            let current_scorer = Scorer {
                symbols: &finder.element_symbols,
                config: &current_config,
                tables,
                element_masses: &finder.element_masses,
            };
            let computed = compute_once(
                finder,
                precursor_mz,
                precursor_formula,
                precursor_ion,
                &peaks,
                &root_counts,
                &current_config,
                &current_scorer,
                solve_options.clone(),
                reduce_graph,
                electron_mass,
            )?;
            let is_high_quality = is_high_quality_tree(&computed, &current_config);
            last_result = Some((computed, tree_size));
            if is_high_quality {
                break;
            }
            increase += config.tree_size_increase;
            tree_size += config.tree_size_increase;
            if tree_size > config.max_tree_size_score {
                break;
            }
        }

        let Some((mut result, selected_tree_size)) = last_result else {
            return Err("no fragmentation tree could be computed".to_string());
        };
        result.result.tree_size_score = selected_tree_size;
        return Ok(result.result);
    }

    let mut result = compute_once(
        finder,
        precursor_mz,
        precursor_formula,
        precursor_ion,
        &peaks,
        &root_counts,
        &config,
        &scorer,
        solve_options,
        reduce_graph,
        electron_mass,
    )?;
    result.result.tree_size_score = config.tree_size_score;
    Ok(result.result)
}

struct ComputedRawTree {
    result: SpectrumTreeResult,
    generated: HashMap<String, GeneratedCandidate>,
    processed_peak_count: usize,
}

#[allow(clippy::too_many_arguments)]
fn compute_once(
    finder: &StoredFormulaFinder,
    precursor_mz: f64,
    precursor_formula: &str,
    precursor_ion: &str,
    peaks: &[SpectrumPeak],
    root_counts: &[i32],
    config: &SiriusLikeConfig,
    scorer: &Scorer<'_>,
    solve_options: TreeSolveOptions,
    reduce_graph: bool,
    electron_mass: f64,
) -> Result<ComputedRawTree, String> {
    let (root_peak, fragment_peaks) = split_root_peak(precursor_mz, peaks, config);
    let processed_peak_count = fragment_peaks.len() + 1;
    let intensity_scale = processed_intensity_scale(root_peak.as_ref(), &fragment_peaks);
    let root_observed_mz = root_peak
        .as_ref()
        .map(|peak| peak.mz)
        .unwrap_or(precursor_mz);
    let root_theoretical_mz = scorer.neutral_mass_from_formula_like_molmass(precursor_formula)?
        + (scorer.element_mass("H")? - electron_mass);
    let root_score = scorer.root_score(
        precursor_formula,
        root_counts,
        if root_peak.is_some() {
            root_observed_mz
        } else {
            root_theoretical_mz
        },
        root_theoretical_mz,
        config.ms2_tolerance_ppm,
    )?;
    let root_candidate = FragmentCandidate {
        formula: precursor_formula.to_string(),
        counts: root_counts.to_vec(),
        ionization: precursor_ion.to_string(),
        peak_id: 0,
        color: 0,
        mass: root_observed_mz,
        score: root_score,
    };
    let root_generated = GeneratedCandidate {
        candidate: root_candidate.clone(),
        theoretical_mz: root_theoretical_mz,
        intensity: root_peak.as_ref().map(|peak| peak.intensity),
    };

    let generated = generate_fragment_candidates(
        finder,
        &fragment_peaks,
        root_counts,
        config,
        scorer,
        precursor_formula,
        precursor_ion,
        intensity_scale,
        electron_mass,
    )?;
    let scoring = build_scoring(&root_generated, &generated, scorer, intensity_scale)?;
    let input = SubFormulaGraphInput {
        root_candidates: vec![root_candidate],
        fragment_candidates: generated
            .values()
            .map(|item| item.candidate.clone())
            .collect(),
        allowed_ionizations: vec![precursor_ion.to_string()],
    };

    let computation = compute_fragmentation_tree(input, &scoring, solve_options, reduce_graph)
        .map_err(|err| err.to_string())?;
    let tree_graph = computation
        .reduced_graph
        .as_ref()
        .unwrap_or(&computation.graph);
    let (reduced_vertex_count, reduced_edge_count) = computation
        .reduced_graph
        .as_ref()
        .map(|graph| (graph.fragments.len(), graph.edges.len()))
        .unwrap_or((0, 0));

    let mut by_formula = generated.clone();
    by_formula.insert(precursor_formula.to_string(), root_generated);
    let fragments = computation
        .tree
        .selected_fragments
        .iter()
        .map(|fragment_id| {
            let vertex = &tree_graph.fragments[*fragment_id];
            let generated = by_formula.get(&vertex.formula).ok_or_else(|| {
                format!("selected formula '{}' was not generated", vertex.formula)
            })?;
            Ok(SelectedRawFragment {
                formula: vertex.formula.clone(),
                counts: vertex.counts.clone(),
                ionization: vertex.ionization.clone(),
                peak_id: vertex.peak_id,
                color: vertex.color,
                mass: vertex.mass,
                score: generated.candidate.score,
                intensity: generated.intensity,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    let losses = computation
        .tree
        .selected_edges
        .iter()
        .filter_map(|edge_id| {
            let edge = &tree_graph.edges[*edge_id];
            if tree_graph.fragments[edge.source].formula.is_empty() {
                return None;
            }
            Some(SelectedRawLoss {
                source_formula: tree_graph.fragments[edge.source].formula.clone(),
                target_formula: tree_graph.fragments[edge.target].formula.clone(),
                score: edge.weight,
            })
        })
        .collect();

    let result = SpectrumTreeResult {
        tree_score: computation.tree.tree_weight,
        is_optimal: computation.tree.is_optimal,
        solver_status: format!("{:?}", computation.tree.status),
        root_formula: tree_graph.fragments[computation.tree.root_fragment]
            .formula
            .clone(),
        fragments,
        losses,
        graph_vertex_count: computation.graph.fragments.len(),
        graph_edge_count: computation.graph.edges.len(),
        reduced_vertex_count,
        reduced_edge_count,
        tree_size_score: config.tree_size_score,
    };

    Ok(ComputedRawTree {
        result,
        generated,
        processed_peak_count,
    })
}

fn is_high_quality_tree(computed: &ComputedRawTree, config: &SiriusLikeConfig) -> bool {
    let mut explainable_by_color: HashMap<i32, f64> = HashMap::new();
    for item in computed.generated.values() {
        let intensity = item.intensity.unwrap_or(0.0);
        explainable_by_color
            .entry(item.candidate.color)
            .and_modify(|current| *current = current.max(intensity))
            .or_insert(intensity);
    }
    if explainable_by_color.is_empty() {
        return false;
    }
    let explained_colors: HashSet<i32> = computed
        .result
        .fragments
        .iter()
        .filter_map(|fragment| {
            if explainable_by_color.contains_key(&fragment.color) {
                Some(fragment.color)
            } else {
                None
            }
        })
        .collect();
    let total_intensity: f64 = explainable_by_color.values().sum();
    let explained_intensity: f64 = explained_colors
        .iter()
        .filter_map(|color| explainable_by_color.get(color))
        .sum();
    let intensity_ratio = if total_intensity <= 0.0 {
        0.0
    } else {
        explained_intensity / total_intensity
    };
    let min_vertices = if config.use_sirius_tree_size_quality_threshold {
        computed
            .processed_peak_count
            .saturating_sub(2)
            .min(config.min_explained_peaks)
    } else {
        (explainable_by_color.len() + 1).min(config.min_explained_peaks)
    };
    intensity_ratio >= config.min_explained_intensity
        && computed.result.fragments.len() >= min_vertices
}

fn split_root_peak(
    precursor_mz: f64,
    peaks: &[SpectrumPeak],
    config: &SiriusLikeConfig,
) -> (Option<ProcessedPeak>, Vec<ProcessedPeak>) {
    let mut processed: Vec<ProcessedPeak> = peaks
        .iter()
        .filter(|peak| peak.intensity > 0.0)
        .map(|peak| ProcessedPeak {
            mz: peak.mz,
            intensity: peak.intensity,
        })
        .collect();
    processed.sort_by(|left, right| partial_cmp(left.mz, right.mz));
    if processed.is_empty() {
        return (None, Vec::new());
    }
    if config.merge_close_peaks {
        processed = remove_close_lower_intensity_peaks(&processed, config);
    }

    let parent_merge_tolerance = (precursor_mz * config.ms2_tolerance_ppm * 2e-6)
        .max(config.mass_deviation_absolute_da * 2.0);
    let parent_indices: Vec<usize> = processed
        .iter()
        .enumerate()
        .filter_map(|(idx, peak)| {
            if (peak.mz - precursor_mz).abs() <= parent_merge_tolerance {
                Some(idx)
            } else {
                None
            }
        })
        .collect();
    let parent_index_set: HashSet<usize> = parent_indices.iter().copied().collect();
    let root_peak = if parent_indices.is_empty() {
        None
    } else {
        let max_parent_intensity = parent_indices
            .iter()
            .map(|idx| processed[*idx].intensity)
            .fold(0.0_f64, f64::max);
        let threshold = max_parent_intensity * 0.1;
        parent_indices
            .iter()
            .filter(|idx| processed[**idx].intensity >= threshold)
            .min_by(|left, right| {
                partial_cmp(
                    (processed[**left].mz - precursor_mz).abs(),
                    (processed[**right].mz - precursor_mz).abs(),
                )
            })
            .map(|idx| processed[*idx].clone())
    };

    let mut fragments: Vec<ProcessedPeak> = processed
        .iter()
        .enumerate()
        .filter_map(|(idx, peak)| {
            if !parent_index_set.contains(&idx) && peak.mz + 0.1 < precursor_mz {
                Some(peak.clone())
            } else {
                None
            }
        })
        .collect();
    let max_intensity = fragments
        .iter()
        .map(|peak| peak.intensity)
        .fold(0.0_f64, f64::max);
    if max_intensity > 0.0 && config.min_relative_intensity > 0.0 {
        fragments.retain(|peak| peak.intensity / max_intensity >= config.min_relative_intensity);
    }

    let mut cap_pool: Vec<(ProcessedPeak, bool)> = fragments
        .iter()
        .cloned()
        .map(|peak| (peak, false))
        .collect();
    if let Some(root) = root_peak.clone() {
        cap_pool.push((root, true));
    }
    if cap_pool.len() > config.max_fragment_peaks {
        cap_pool.sort_by(|left, right| {
            partial_cmp(right.0.intensity, left.0.intensity)
                .then_with(|| partial_cmp(left.0.mz, right.0.mz))
        });
        fragments = cap_pool
            .into_iter()
            .take(config.max_fragment_peaks)
            .filter_map(|(peak, is_root)| if is_root { None } else { Some(peak) })
            .collect();
        fragments.sort_by(|left, right| partial_cmp(left.mz, right.mz));
    }
    (root_peak, fragments)
}

fn remove_close_lower_intensity_peaks(
    peaks: &[ProcessedPeak],
    config: &SiriusLikeConfig,
) -> Vec<ProcessedPeak> {
    let mut mass_sorted = peaks.to_vec();
    mass_sorted.sort_by(|left, right| partial_cmp(left.mz, right.mz));
    let mut deleted = vec![false; mass_sorted.len()];
    let mut intensity_order: Vec<usize> = (0..mass_sorted.len()).collect();
    intensity_order.sort_by(|left, right| {
        partial_cmp(mass_sorted[*right].intensity, mass_sorted[*left].intensity)
            .then_with(|| partial_cmp(mass_sorted[*left].mz, mass_sorted[*right].mz))
    });

    for index in intensity_order {
        if deleted[index] {
            continue;
        }
        let center_mz = mass_sorted[index].mz;
        let mut left = index;
        while left > 0 {
            left -= 1;
            if !in_doubled_ms2_window(center_mz, mass_sorted[left].mz, config) {
                break;
            }
            deleted[left] = true;
        }
        let mut right = index + 1;
        while right < mass_sorted.len()
            && in_doubled_ms2_window(center_mz, mass_sorted[right].mz, config)
        {
            deleted[right] = true;
            right += 1;
        }
    }

    mass_sorted
        .into_iter()
        .enumerate()
        .filter_map(|(idx, peak)| if deleted[idx] { None } else { Some(peak) })
        .collect()
}

fn in_doubled_ms2_window(center_mz: f64, mz: f64, config: &SiriusLikeConfig) -> bool {
    let window =
        (center_mz * config.ms2_tolerance_ppm * 2e-6).max(config.mass_deviation_absolute_da * 2.0);
    (mz - center_mz).abs() <= window
}

fn processed_intensity_scale(
    root_peak: Option<&ProcessedPeak>,
    fragment_peaks: &[ProcessedPeak],
) -> f64 {
    let mut max_intensity = fragment_peaks
        .iter()
        .map(|peak| peak.intensity)
        .fold(0.0_f64, f64::max);
    if let Some(root) = root_peak {
        max_intensity = max_intensity.max(root.intensity);
    }
    if max_intensity > 0.0 {
        max_intensity
    } else {
        1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_fragment_candidates(
    finder: &StoredFormulaFinder,
    peaks: &[ProcessedPeak],
    root_counts: &[i32],
    config: &SiriusLikeConfig,
    scorer: &Scorer<'_>,
    root_formula: &str,
    ionization: &str,
    intensity_scale: f64,
    electron_mass: f64,
) -> Result<HashMap<String, GeneratedCandidate>, String> {
    let mut max_count_symbols = Vec::new();
    let mut max_count_values = Vec::new();
    for (symbol, count) in finder.element_symbols.iter().zip(root_counts.iter()) {
        if *count > 0 {
            max_count_symbols.push(symbol.clone());
            max_count_values.push(*count as f64);
        }
    }
    let mut candidates_by_peak: Vec<HashMap<String, GeneratedCandidate>> = Vec::new();

    for (peak_idx, peak) in peaks.iter().enumerate() {
        let output = finder.find_formulae_public(
            peak.mz,
            1,
            config.candidate_search_ppm,
            config.candidate_search_absolute_da,
            None,
            Vec::new(),
            Vec::new(),
            None,
            max_count_symbols.clone(),
            max_count_values.clone(),
            config.candidate_limit_per_peak,
            Some((-1.5, 80.0)),
            false,
            Some("H".to_string()),
            false,
            Vec::new(),
            Vec::new(),
            0.0,
            0.0,
            electron_mass,
            false,
            Vec::new(),
            Vec::new(),
            0.0,
            0.0,
            0.0,
            0.0,
        )?;
        let relative_intensity = if intensity_scale <= 0.0 {
            0.0
        } else {
            peak.intensity / intensity_scale
        };
        let mut peak_generated = HashMap::new();
        for idx in 0..output.output.counts.len() {
            let counts = output.output.counts.row(idx).to_vec();
            let formula = formula::format_formula_from_counts(
                &output.output.core_symbols,
                &counts.iter().map(|count| *count as i64).collect::<Vec<_>>(),
                output.output.formula_charge,
            );
            if formula == root_formula {
                continue;
            }
            if !is_subformula_counts(root_counts, &counts) {
                continue;
            }
            let theoretical_mz = peak.mz + output.output.error_da[idx];
            let allowed_delta =
                (peak.mz * config.ms2_tolerance_ppm * 1e-6).max(config.mass_deviation_absolute_da);
            if (peak.mz - theoretical_mz).abs() > allowed_delta {
                continue;
            }
            let candidate_score = scorer.fragment_candidate_score(
                &formula,
                &counts,
                peak.mz,
                theoretical_mz,
                output.output.exact_masses[idx],
                relative_intensity,
            )?;
            let candidate = FragmentCandidate {
                formula: formula.clone(),
                counts: counts.clone(),
                ionization: ionization.to_string(),
                peak_id: peak_idx + 1,
                color: (peak_idx + 1) as i32,
                mass: peak.mz,
                score: candidate_score,
            };
            peak_generated.insert(
                formula,
                GeneratedCandidate {
                    candidate,
                    theoretical_mz,
                    intensity: Some(peak.intensity),
                },
            );
        }
        candidates_by_peak.push(peak_generated);
    }

    disjoin_nearby_fragment_candidates(peaks, &mut candidates_by_peak, config);

    let mut generated = HashMap::new();
    for peak_generated in candidates_by_peak {
        for (formula, item) in peak_generated {
            let replace = generated
                .get(&formula)
                .map(|previous| candidate_mass_error(&item) < candidate_mass_error(previous))
                .unwrap_or(true);
            if replace {
                generated.insert(formula, item);
            }
        }
    }
    Ok(generated)
}

fn disjoin_nearby_fragment_candidates(
    peaks: &[ProcessedPeak],
    candidates_by_peak: &mut [HashMap<String, GeneratedCandidate>],
    config: &SiriusLikeConfig,
) {
    for index in 1..peaks.len() {
        if !in_doubled_ms2_window(peaks[index].mz, peaks[index - 1].mz, config) {
            continue;
        }
        let common: Vec<String> = candidates_by_peak[index - 1]
            .keys()
            .filter(|formula| candidates_by_peak[index].contains_key(*formula))
            .cloned()
            .collect();
        for formula in common {
            let left_error = candidates_by_peak[index - 1]
                .get(&formula)
                .map(candidate_mass_error)
                .unwrap_or(f64::INFINITY);
            let right_error = candidates_by_peak[index]
                .get(&formula)
                .map(candidate_mass_error)
                .unwrap_or(f64::INFINITY);
            if left_error < right_error {
                candidates_by_peak[index].remove(&formula);
            } else {
                candidates_by_peak[index - 1].remove(&formula);
            }
        }
    }
}

fn candidate_mass_error(item: &GeneratedCandidate) -> f64 {
    (item.candidate.mass - item.theoretical_mz).abs()
}

fn build_scoring(
    root: &GeneratedCandidate,
    generated: &HashMap<String, GeneratedCandidate>,
    scorer: &Scorer<'_>,
    intensity_scale: f64,
) -> Result<GraphScoring, String> {
    let mut peak_scores = BTreeMap::new();
    for item in generated.values() {
        let relative_intensity = if intensity_scale <= 0.0 {
            0.0
        } else {
            item.intensity.unwrap_or(0.0) / intensity_scale
        };
        peak_scores.insert(
            item.candidate.color,
            scorer.peak_score(item.candidate.mass, relative_intensity),
        );
    }

    let mut fragment_scores = BTreeMap::new();
    for (formula, item) in generated {
        if !is_subformula_counts(&root.candidate.counts, &item.candidate.counts) {
            continue;
        }
        let root_loss_counts = subtract_counts(&root.candidate.counts, &item.candidate.counts)?;
        let root_loss_formula = format_counts(scorer.symbols, &root_loss_counts);
        fragment_scores.insert(
            formula.clone(),
            scorer.common_root_loss_score(&root_loss_formula)
                + scorer.db_paired_score(formula, Some(&root.candidate.formula)),
        );
    }

    let mut by_color: Vec<&GeneratedCandidate> = generated.values().collect();
    by_color.sort_by_key(|item| item.candidate.color);
    let mut peak_pair_scores = BTreeMap::new();
    for child in &by_color {
        if root.candidate.mass > child.candidate.mass {
            peak_pair_scores.insert(
                (root.candidate.color, child.candidate.color),
                scorer.peak_pair_score(root.candidate.mass, child.candidate.mass),
            );
        }
    }
    for parent in &by_color {
        for child in &by_color {
            if parent.candidate.mass > child.candidate.mass {
                peak_pair_scores.insert(
                    (parent.candidate.color, child.candidate.color),
                    scorer.peak_pair_score(parent.candidate.mass, child.candidate.mass),
                );
            }
        }
    }

    let mut loss_scores = BTreeMap::new();
    for (child_formula, child) in generated {
        if root.candidate.mass <= child.candidate.mass
            || !is_subformula_counts(&root.candidate.counts, &child.candidate.counts)
        {
            continue;
        }
        let loss_counts = subtract_counts(&root.candidate.counts, &child.candidate.counts)?;
        let loss_formula = format_counts(scorer.symbols, &loss_counts);
        loss_scores.insert(
            (root.candidate.formula.clone(), child_formula.clone()),
            scorer.loss_score(
                &loss_formula,
                &loss_counts,
                root.candidate.mass - child.candidate.mass,
                root.theoretical_mz - child.theoretical_mz,
                Some(child_formula),
                true,
            )?,
        );
    }
    for (parent_formula, parent) in generated {
        for (child_formula, child) in generated {
            if parent_formula == child_formula
                || parent.candidate.mass <= child.candidate.mass
                || !is_subformula_counts(&parent.candidate.counts, &child.candidate.counts)
            {
                continue;
            }
            let loss_counts = subtract_counts(&parent.candidate.counts, &child.candidate.counts)?;
            let loss_formula = format_counts(scorer.symbols, &loss_counts);
            loss_scores.insert(
                (parent_formula.clone(), child_formula.clone()),
                scorer.loss_score(
                    &loss_formula,
                    &loss_counts,
                    parent.candidate.mass - child.candidate.mass,
                    parent.theoretical_mz - child.theoretical_mz,
                    Some(child_formula),
                    false,
                )?,
            );
        }
    }

    Ok(GraphScoring {
        peak_scores,
        peak_pair_scores,
        fragment_scores,
        loss_scores,
        general_graph_score: 0.0,
    })
}

impl<'a> Scorer<'a> {
    fn root_score(
        &self,
        formula: &str,
        counts: &[i32],
        observed_mz: f64,
        theoretical_mz: f64,
        ppm: f64,
    ) -> Result<f64, String> {
        Ok(self.mass_deviation_score(
            observed_mz,
            theoretical_mz,
            ppm,
            self.config.mass_deviation_vertex_weight,
            self.config.mass_deviation_absolute_da,
        ) + self.intrinsically_charged_root_score(counts)
            + self.phosphor_root_score(counts)
            + self.strange_element_root_score(counts)
            + self.config.chemical_prior_root_score
            + self.carbohydrogen_root_score(counts)
            + self.db_paired_score(formula, None))
    }

    fn fragment_candidate_score(
        &self,
        formula: &str,
        counts: &[i32],
        observed_mz: f64,
        theoretical_mz: f64,
        neutral_mass: f64,
        relative_intensity: f64,
    ) -> Result<f64, String> {
        Ok(self.mass_deviation_score(
            observed_mz,
            theoretical_mz,
            self.config.ms2_tolerance_ppm,
            self.config.mass_deviation_vertex_weight,
            self.config.mass_deviation_absolute_da,
        ) + self.phosphor_fragment_score(counts)
            + self.strange_element_small_fragment_score(counts, neutral_mass)
            + self.strange_element_fragment_score(formula, counts, neutral_mass)
            + self.common_fragment_score(formula)
            + self.carbohydrogen_fragment_score(counts, relative_intensity))
    }

    fn peak_score(&self, mz: f64, relative_intensity: f64) -> f64 {
        self.clipped_peak_is_noise_score(relative_intensity)
            + self.config.tree_size_score
            + self.fragment_size_score(mz)
    }

    fn peak_pair_score(&self, parent_mz: f64, child_mz: f64) -> f64 {
        let delta = parent_mz - child_mz;
        if delta <= 0.0 {
            0.0
        } else {
            self.loss_size_score(delta)
        }
    }

    fn loss_score(
        &self,
        loss_formula: &str,
        loss_counts: &[i32],
        observed_delta: f64,
        theoretical_delta: f64,
        child_formula: Option<&str>,
        is_root_loss: bool,
    ) -> Result<f64, String> {
        Ok(self.mass_deviation_score(
            observed_delta,
            theoretical_delta,
            self.config.ms2_tolerance_ppm,
            self.config.mass_deviation_edge_weight,
            self.config.loss_mass_deviation_absolute_da,
        ) + self.phosphor_fragment_score(loss_counts)
            + self.free_radical_loss_score(loss_formula, loss_counts)
            + self.dbe_loss_score(loss_counts)
            + self.pure_carbon_nitrogen_loss_score(loss_counts)
            + self.strange_element_loss_score(loss_formula)
            + self.common_loss_score(loss_formula)
            + self.multimere_loss_score(loss_formula, child_formula, is_root_loss)
            + self.fatty_acid_chain_loss_score(loss_formula, loss_counts))
    }

    fn mass_deviation_score(
        &self,
        observed_mz: f64,
        theoretical_mz: f64,
        ppm: f64,
        weight: f64,
        absolute_da: f64,
    ) -> f64 {
        let sigma = (observed_mz.abs() * ppm * 1e-6).max(absolute_da);
        if sigma <= 0.0 || !sigma.is_finite() {
            return -100.0;
        }
        let x = (observed_mz - theoretical_mz).abs() / (std::f64::consts::SQRT_2 * sigma);
        let prob = unsafe { c_erfc(x) }.max(1e-300);
        (weight * unsafe { c_log(prob) }).max(-100.0)
    }

    fn clipped_peak_is_noise_score(&self, relative_intensity: f64) -> f64 {
        if relative_intensity <= 0.0 {
            return 0.0;
        }
        let cdf_one = self.pareto_cdf(1.0);
        let c = 1.0 - cdf_one;
        let q = (1.0 - self.pareto_cdf(relative_intensity.min(1.0)) - c
            + self.config.clipped_noise_beta)
            / (1.0 - c + self.config.clipped_noise_beta);
        -q.max(1e-300).ln()
    }

    fn fragment_size_score(&self, mz: f64) -> f64 {
        let fraction = (mz.max(0.0) / self.config.fragment_size_max_mz).clamp(0.0, 1.0);
        self.config.fragment_size_max_score * (1.0 - fraction)
    }

    fn loss_size_score(&self, mass: f64) -> f64 {
        if mass <= 0.0 {
            return -100.0;
        }
        let variance = self.config.loss_size_variance;
        let sd = variance.sqrt();
        let density = (-((mass.ln() - self.config.loss_size_mean).powi(2)) / (2.0 * variance))
            .exp()
            / ((2.0 * std::f64::consts::PI).sqrt() * sd * mass);
        density.max(1e-12).ln() - self.config.loss_size_normalization
    }

    fn intrinsically_charged_root_score(&self, counts: &[i32]) -> f64 {
        if self.maybe_charged_counts(counts) {
            self.config.intrinsically_charged_root_penalty
        } else {
            0.0
        }
    }

    fn phosphor_root_score(&self, counts: &[i32]) -> f64 {
        let p = self.count_of(counts, "P");
        if p > 0 && self.count_of(counts, "O") + self.count_of(counts, "S") < 2 * p {
            0.05_f64.ln()
        } else {
            0.0
        }
    }

    fn phosphor_fragment_score(&self, counts: &[i32]) -> f64 {
        let p = self.count_of(counts, "P");
        if p > 0 && self.count_of(counts, "O") < p && self.count_of(counts, "S") < p {
            0.25_f64.ln()
        } else {
            0.0
        }
    }

    fn strange_element_root_score(&self, counts: &[i32]) -> f64 {
        let n_strange = self
            .symbols
            .iter()
            .zip(counts.iter())
            .filter(|(symbol, count)| {
                **count > 0 && !matches!(symbol.as_str(), "C" | "H" | "N" | "O")
            })
            .count();
        self.config.strange_element_root_penalty * (n_strange as f64)
    }

    fn strange_element_small_fragment_score(&self, counts: &[i32], neutral_mass: f64) -> f64 {
        if neutral_mass > self.config.strange_element_small_fragment_max_mass {
            return 0.0;
        }
        if self
            .symbols
            .iter()
            .zip(counts.iter())
            .any(|(symbol, count)| *count > 0 && !matches!(symbol.as_str(), "C" | "H" | "N" | "O"))
        {
            self.config.strange_element_small_fragment_score
        } else {
            0.0
        }
    }

    fn strange_element_fragment_score(
        &self,
        formula: &str,
        counts: &[i32],
        neutral_mass: f64,
    ) -> f64 {
        if self.tables.strange_fragment_whitelist.contains(formula) {
            return self.config.strange_element_fragment_score;
        }
        if neutral_mass < self.config.strange_element_fragment_min_mass {
            return 0.0;
        }
        if self
            .symbols
            .iter()
            .zip(counts.iter())
            .any(|(symbol, count)| *count > 0 && !matches!(symbol.as_str(), "C" | "H" | "N" | "O"))
        {
            self.config.strange_element_fragment_penalty
        } else {
            0.0
        }
    }

    fn common_fragment_score(&self, formula: &str) -> f64 {
        if !self.config.enable_common_fragment_score {
            return 0.0;
        }
        self.tables
            .common_fragments
            .get(formula)
            .map(|score| *score - self.tables.common_fragment_normalization)
            .unwrap_or(0.0)
    }

    fn free_radical_loss_score(&self, formula: &str, counts: &[i32]) -> f64 {
        if let Some(score) = self.tables.common_radicals.get(formula) {
            return *score - self.config.free_radical_normalization;
        }
        if self.maybe_charged_counts(counts) {
            self.config.free_radical_penalty - self.config.free_radical_normalization
        } else {
            -self.config.free_radical_normalization
        }
    }

    fn dbe_loss_score(&self, counts: &[i32]) -> f64 {
        if let Some(doubled) = self.doubled_rdbe(counts) {
            if doubled < 0 {
                return 0.05_f64
                    .ln()
                    .max((doubled.abs() as f64) * self.config.dbe_loss_score);
            }
        }
        0.0
    }

    fn pure_carbon_nitrogen_loss_score(&self, counts: &[i32]) -> f64 {
        let total: i32 = counts.iter().sum();
        if total <= 0 {
            return 0.0;
        }
        let cn = self.count_of(counts, "C") + self.count_of(counts, "N");
        if cn >= total {
            self.config.pure_carbon_nitrogen_loss_penalty
        } else {
            0.0
        }
    }

    fn strange_element_loss_score(&self, formula: &str) -> f64 {
        if self.tables.strange_losses.contains(formula) {
            self.config.strange_element_loss_score
        } else {
            0.0
        }
    }

    fn common_loss_score(&self, formula: &str) -> f64 {
        if self.config.recombine_common_losses {
            if let Some(score) = self.tables.recombined_common_loss_overrides.get(formula) {
                return *score;
            }
            if let Some(score) = self.tables.recombined_common_losses.get(formula) {
                if *score != 0.0 {
                    return *score - self.tables.common_loss_normalization;
                }
            }
        }
        self.tables
            .common_losses
            .get(formula)
            .map(|score| *score - self.tables.common_loss_normalization)
            .unwrap_or(-self.tables.common_loss_normalization)
    }

    fn common_root_loss_score(&self, formula: &str) -> f64 {
        self.tables
            .common_root_losses
            .get(formula)
            .map(|score| *score - self.tables.common_root_loss_normalization)
            .unwrap_or(-self.tables.common_root_loss_normalization)
    }

    fn db_paired_score(&self, formula: &str, root_formula: Option<&str>) -> f64 {
        let Some(formulas) = self.config.db_paired_formulas.as_ref() else {
            return 0.0;
        };
        if !formulas.contains(formula) {
            return 0.0;
        }
        if let Some(root) = root_formula {
            if !formulas.contains(root) {
                return 0.0;
            }
        }
        self.config.db_paired_formula_score
    }

    fn multimere_loss_score(
        &self,
        loss_formula: &str,
        child_formula: Option<&str>,
        is_root_loss: bool,
    ) -> f64 {
        let Some(child) = child_formula else {
            return 0.0;
        };
        if loss_formula != child {
            return 0.0;
        }
        if is_root_loss {
            self.config.multimere_root_loss_score
        } else {
            self.config.multimere_loss_score
        }
    }

    fn fatty_acid_chain_loss_score(&self, formula: &str, counts: &[i32]) -> f64 {
        let Some((chain_length, double_bonds)) = self.lipid_chain_from_counts(counts) else {
            return 0.0;
        };
        if chain_length < self.config.fatty_acid_chain_min_length
            || chain_length > self.config.fatty_acid_chain_max_length
            || double_bonds > self.config.fatty_acid_chain_max_double_bonds
        {
            return 0.0;
        }
        let penalty = self.loss_size_score(
            self.neutral_mass_from_formula_like_molmass(formula)
                .unwrap_or(0.0),
        );
        if penalty >= 0.0 {
            return 0.0;
        }
        -penalty
            * self.config.fatty_acid_chain_score_weight
            * self
                .config
                .fatty_acid_chain_double_bond_decay
                .powi(double_bonds * double_bonds)
    }

    fn carbohydrogen_root_score(&self, counts: &[i32]) -> f64 {
        if self.is_cho_counts(counts) {
            self.config.carbohydrogen_root_score
        } else {
            0.0
        }
    }

    fn carbohydrogen_fragment_score(&self, counts: &[i32], relative_intensity: f64) -> f64 {
        if !self.config.enable_carbohydrogen_fragment_score
            || relative_intensity <= self.config.carbohydrogen_fragment_min_relative_intensity
            || !self.is_cho_counts(counts)
        {
            return 0.0;
        }
        pareto_cdf_from_median(
            relative_intensity,
            self.config.carbohydrogen_fragment_xmin,
            self.config.carbohydrogen_fragment_median,
        )
    }

    fn count_of(&self, counts: &[i32], symbol: &str) -> i32 {
        self.symbols
            .iter()
            .position(|item| item == symbol)
            .map(|idx| counts[idx])
            .unwrap_or(0)
    }

    fn is_cho_counts(&self, counts: &[i32]) -> bool {
        self.symbols
            .iter()
            .zip(counts.iter())
            .all(|(symbol, count)| *count <= 0 || matches!(symbol.as_str(), "C" | "H" | "O"))
    }

    fn doubled_rdbe(&self, counts: &[i32]) -> Option<i32> {
        let mut total = 2;
        for (symbol, count) in self.symbols.iter().zip(counts.iter()) {
            let valence = chemistry::bond_electrons(symbol)?;
            total += count * (valence - 2);
        }
        Some(total)
    }

    fn maybe_charged_counts(&self, counts: &[i32]) -> bool {
        let Some(doubled) = self.doubled_rdbe(counts) else {
            return false;
        };
        if self.config.strict_sirius_radical_parity {
            doubled > 0 && doubled % 2 == 1
        } else {
            doubled.abs() % 2 == 1
        }
    }

    fn lipid_chain_from_counts(&self, counts: &[i32]) -> Option<(i32, i32)> {
        if self
            .symbols
            .iter()
            .zip(counts.iter())
            .any(|(symbol, count)| *count > 0 && !matches!(symbol.as_str(), "C" | "H" | "N" | "O"))
        {
            return None;
        }
        let c = self.count_of(counts, "C");
        let h = self.count_of(counts, "H");
        let n = self.count_of(counts, "N");
        let o = self.count_of(counts, "O");
        if c < 2 {
            return None;
        }
        if n > 0 {
            if n == 1 && o == 2 && h % 2 != 0 {
                let double_bonds = ((c * 2 + 3) - h) / 2;
                if double_bonds >= c / 2 {
                    return None;
                }
                if double_bonds >= 0 {
                    return Some((c, double_bonds));
                }
            }
        } else if o > 0 && h % 2 == 0 {
            if o == 1 {
                let double_bonds = ((c * 2 - 2) - h) / 2;
                if double_bonds >= c / 2 {
                    return None;
                }
                if double_bonds >= 0 {
                    return Some((c, double_bonds));
                }
            }
        } else if h % 2 == 0 {
            let double_bonds = (2 * c - h) / 2;
            if double_bonds >= c / 2 {
                return None;
            }
            if double_bonds >= 0 {
                return Some((c, double_bonds));
            }
        }
        None
    }

    fn pareto_cdf(&self, x: f64) -> f64 {
        if x < self.config.clipped_noise_xmin {
            return 0.0;
        }
        let median = self
            .config
            .median_noise_intensity
            .max(self.config.clipped_noise_xmin * (1.0 + 1e-9));
        let k = 2.0_f64.ln() / (median / self.config.clipped_noise_xmin).ln();
        1.0 - (self.config.clipped_noise_xmin / x).powf(k)
    }

    fn element_mass(&self, symbol: &str) -> Result<f64, String> {
        self.element_masses
            .get(symbol)
            .copied()
            .ok_or_else(|| format!("element '{symbol}' is not in the mass table"))
    }

    fn neutral_mass_from_formula_like_molmass(&self, formula: &str) -> Result<f64, String> {
        let symbols = formula::parse_element_symbols(formula)?;
        let counts = formula::parse_formula_counts(formula, &symbols)?;
        let mut mass = 0.0;
        for (symbol, count) in symbols.iter().zip(counts.iter()).rev() {
            mass += self.element_mass(symbol)? * (*count as f64);
        }
        Ok(mass)
    }
}

fn pareto_cdf_from_median(x: f64, xmin: f64, median: f64) -> f64 {
    if x < xmin {
        return 0.0;
    }
    let k = 2.0_f64.ln() / (median / xmin).ln();
    1.0 - (xmin / x).powf(k)
}

fn counts_i64_to_i32(counts: &[i64]) -> Result<Vec<i32>, String> {
    counts
        .iter()
        .map(|count| {
            i32::try_from(*count)
                .map_err(|_| format!("formula count {count} is outside the supported range"))
        })
        .collect()
}

fn is_subformula_counts(parent: &[i32], child: &[i32]) -> bool {
    parent.len() == child.len()
        && parent
            .iter()
            .zip(child.iter())
            .all(|(parent_count, child_count)| parent_count >= child_count)
}

fn subtract_counts(parent: &[i32], child: &[i32]) -> Result<Vec<i32>, String> {
    if !is_subformula_counts(parent, child) {
        return Err("child counts must be a subformula of parent counts".to_string());
    }
    Ok(parent
        .iter()
        .zip(child.iter())
        .map(|(parent_count, child_count)| parent_count - child_count)
        .collect())
}

fn format_counts(symbols: &[String], counts: &[i32]) -> String {
    formula::format_formula_from_counts(
        symbols,
        &counts.iter().map(|count| *count as i64).collect::<Vec<_>>(),
        0,
    )
}

fn partial_cmp(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}
