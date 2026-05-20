use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct PriorElementModel {
    pub p_absent: f64,
    pub points: Vec<f64>,
    pub weights: Vec<f64>,
    pub variance: f64,
}

#[derive(Clone, Debug)]
pub struct PriorScorer {
    c_index: Option<usize>,
    ratio_indices: Vec<Option<usize>>,
    models: Vec<PriorElementModel>,
    uniform_weight: f64,
}

impl PriorScorer {
    pub fn new(
        core_symbols: &[String],
        ratio_elements: &[String],
        p_absent: &[f64],
        kde_points: &[Vec<f64>],
        kde_weights: &[Vec<f64>],
        kde_variance: &[f64],
        uniform_weight: f64,
    ) -> Result<Self, String> {
        if ratio_elements.len() != p_absent.len()
            || ratio_elements.len() != kde_points.len()
            || ratio_elements.len() != kde_weights.len()
            || ratio_elements.len() != kde_variance.len()
        {
            return Err("prior payload arrays must have matching lengths".to_string());
        }

        let mut models = Vec::with_capacity(ratio_elements.len());
        for idx in 0..ratio_elements.len() {
            if kde_points[idx].len() != kde_weights[idx].len() {
                return Err("KDE point and weight arrays must have matching lengths".to_string());
            }
            models.push(PriorElementModel {
                p_absent: p_absent[idx],
                points: kde_points[idx].clone(),
                weights: kde_weights[idx].clone(),
                variance: kde_variance[idx],
            });
        }

        Ok(Self {
            c_index: core_symbols.iter().position(|symbol| symbol == "C"),
            ratio_indices: ratio_elements
                .iter()
                .map(|element| core_symbols.iter().position(|symbol| symbol == element))
                .collect(),
            models,
            uniform_weight,
        })
    }

    pub fn score_counts(&self, counts: &[i32]) -> f64 {
        let Some(c_index) = self.c_index else {
            return 0.0;
        };
        let c_count = counts.get(c_index).copied().unwrap_or(0);
        if c_count == 0 {
            return 0.0;
        }

        let c_count = c_count as f64;
        let mut log_p = 0.0;
        for (idx, model) in self.models.iter().enumerate() {
            let count = self.ratio_indices[idx]
                .and_then(|col| counts.get(col).copied())
                .unwrap_or(0);
            if count == 0 {
                log_p += (model.p_absent + self.uniform_weight).ln();
                continue;
            }

            if model.points.is_empty() || model.variance <= 0.0 {
                log_p += self.uniform_weight.ln();
                continue;
            }

            let ratio = count as f64 / c_count;
            let p_present = 1.0 - model.p_absent;
            let kde_val =
                gaussian_kde_density(ratio, &model.points, &model.weights, model.variance);
            log_p += (p_present * kde_val + self.uniform_weight).ln();
        }
        log_p
    }
}

fn gaussian_kde_density(x: f64, points: &[f64], weights: &[f64], variance: f64) -> f64 {
    let normalizer = (2.0 * PI * variance).sqrt();
    let weighted_sum: f64 = points
        .iter()
        .zip(weights.iter())
        .map(|(point, weight)| {
            let delta = x - point;
            weight * (-0.5 * delta * delta / variance).exp()
        })
        .sum();
    weighted_sum / normalizer
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn no_carbon_scores_uninformative_zero() {
        let scorer = PriorScorer::new(
            &strings(&["H", "O"]),
            &strings(&["H", "O"]),
            &[0.0, 0.5],
            &[vec![2.0], vec![1.0]],
            &[vec![1.0], vec![1.0]],
            &[0.1, 0.1],
            1e-6,
        )
        .unwrap();

        assert_eq!(scorer.score_counts(&[2, 1]), 0.0);
    }

    #[test]
    fn absent_ratio_element_uses_absent_probability() {
        let scorer = PriorScorer::new(
            &strings(&["C", "H", "O"]),
            &strings(&["H", "O"]),
            &[0.0, 0.25],
            &[vec![2.0], vec![1.0]],
            &[vec![1.0], vec![1.0]],
            &[0.1, 0.1],
            1e-6,
        )
        .unwrap();

        let score = scorer.score_counts(&[6, 12, 0]);
        let h_density = gaussian_kde_density(2.0, &[2.0], &[1.0], 0.1);
        let expected = (h_density + 1e-6).ln() + (0.25_f64 + 1e-6).ln();
        assert!((score - expected).abs() < 1e-12);
    }

    #[test]
    fn present_ratio_element_uses_weighted_kde_density() {
        let scorer = PriorScorer::new(
            &strings(&["C", "H"]),
            &strings(&["H"]),
            &[0.25],
            &[vec![1.0, 3.0]],
            &[vec![0.25, 0.75]],
            &[0.5],
            1e-6,
        )
        .unwrap();

        let score = scorer.score_counts(&[2, 4]);
        let density = gaussian_kde_density(2.0, &[1.0, 3.0], &[0.25, 0.75], 0.5);
        let expected = ((1.0 - 0.25) * density + 1e-6).ln();
        assert!((score - expected).abs() < 1e-12);
    }

    #[test]
    fn invalid_kde_payload_is_rejected() {
        let err = PriorScorer::new(
            &strings(&["C", "H"]),
            &strings(&["H"]),
            &[0.25],
            &[vec![1.0, 3.0]],
            &[vec![1.0]],
            &[0.5],
            1e-6,
        )
        .unwrap_err();

        assert!(err.contains("KDE point and weight arrays"));
    }
}
