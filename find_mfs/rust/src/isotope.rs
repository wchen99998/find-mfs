pub fn within_ratio_tolerance(predicted: f64, observed: f64, tol_rel: f64, tol_abs: f64) -> bool {
    let tol = (tol_rel * observed).max(tol_abs);
    (predicted - observed).abs() <= tol
}
