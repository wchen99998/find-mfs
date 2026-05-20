pub fn approx_m2_from_counts(approx_m1: f64, counts: &[i64], iso_m2_direct: &[f64]) -> f64 {
    let mut approx_m2 = approx_m1 * approx_m1 * 0.5;
    for (count, coeff) in counts.iter().zip(iso_m2_direct.iter()) {
        approx_m2 += (*count as f64) * coeff;
    }
    approx_m2
}

pub fn within_ratio_tolerance(predicted: f64, observed: f64, tol_rel: f64, tol_abs: f64) -> bool {
    let tol = (tol_rel * observed).max(tol_abs);
    (predicted - observed).abs() <= tol
}
