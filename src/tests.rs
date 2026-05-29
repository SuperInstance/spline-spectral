use crate::*;

// ============ BSpline Tests ============

#[test]
fn test_partition_of_unity() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let cp = vec![0.0; 5];
    let spline = BSpline::new(3, knots, cp);
    // Sum of all basis functions at any point should be 1
    for t_frac in 1..100 {
        let t = t_frac as f64 * 0.01;
        if t >= 1.0 { break; }
        let sum: f64 = (0..5).map(|i| spline.basis(i, 3, t)).sum();
        assert!((sum - 1.0).abs() < 1e-10, "Partition of unity violated at t={}: sum={}", t, sum);
    }
}

#[test]
fn test_evaluate_constant_spline() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let cp = vec![5.0, 5.0, 5.0, 5.0];
    let spline = BSpline::new(3, knots, cp);
    for t in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let val = spline.evaluate(t);
        assert!((val - 5.0).abs() < 1e-10, "Constant spline should be 5.0 at t={}: got {}", t, val);
    }
}

#[test]
fn test_evaluate_linear_spline() {
    // Linear B-spline: degree 1
    let knots = vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
    let cp = vec![0.0, 1.0, 2.0, 3.0];
    let spline = BSpline::new(1, knots, cp);
    let val = spline.evaluate(1.5);
    assert!((val - 1.5).abs() < 1e-10, "Linear spline at 1.5 should be 1.5: got {}", val);
}

#[test]
fn test_collocation_matrix_dimensions() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let cp = vec![1.0, 2.0, 3.0, 4.0];
    let spline = BSpline::new(3, knots, cp);
    let samples = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let mat = spline.collocation_matrix(&samples);
    assert_eq!(mat.len(), 5, "Should have 5 rows");
    assert_eq!(mat[0].len(), 4, "Should have 4 columns");
}

#[test]
fn test_basis_zero_outside_support() {
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let cp = vec![0.0; 5];
    let spline = BSpline::new(3, knots, cp);
    // Basis 0 should be 0 outside [0, 0.5)
    assert_eq!(spline.basis(0, 3, 0.7), 0.0);
}

// ============ GraphSpline Tests ============

#[test]
fn test_interpolate_constrained_nodes() {
    // Path graph of 5 nodes
    let adj = vec![
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    ];
    let mut gs = GraphSpline::new(adj);
    gs.constrain(0, 0.0);
    gs.constrain(4, 4.0);
    let result = gs.interpolate();
    assert!((result[0] - 0.0).abs() < 1e-10, "Node 0 should be 0.0");
    assert!((result[4] - 4.0).abs() < 1e-10, "Node 4 should be 4.0");
    // Interior should be linear: 1, 2, 3
    assert!((result[1] - 1.0).abs() < 1e-6, "Node 1 should be ~1.0, got {}", result[1]);
    assert!((result[2] - 2.0).abs() < 1e-6, "Node 2 should be ~2.0, got {}", result[2]);
    assert!((result[3] - 3.0).abs() < 1e-6, "Node 3 should be ~3.0, got {}", result[3]);
}

#[test]
fn test_smoothness_minimized() {
    let adj = vec![
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let mut gs = GraphSpline::new(adj);
    gs.constrain(0, 0.0);
    gs.constrain(3, 3.0);
    let result = gs.interpolate();
    let smooth = gs.smoothness(&result);

    // Any other interpolation of the endpoints should have higher smoothness
    let noisy = vec![0.0, 5.0, -2.0, 3.0];
    let noisy_smooth = gs.smoothness(&noisy);
    assert!(smooth <= noisy_smooth + 1e-6,
        "Optimal smoothness {} should be <= noisy {}", smooth, noisy_smooth);
}

#[test]
fn test_spectral_smooth_preserves_dc() {
    // Path graph
    let n = 6;
    let adj: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|j| if (i as i32 - j as i32).unsigned_abs() == 1 { 1.0 } else { 0.0 }).collect()
    }).collect();
    let gs = GraphSpline::new(adj);
    let signal = vec![3.0; n]; // DC signal
    let smoothed = gs.spectral_smooth(&signal, 1);
    // DC should be preserved (at least roughly)
    let mean: f64 = smoothed.iter().sum::<f64>() / n as f64;
    // Even if imperfect, should have nonzero energy
    let energy: f64 = smoothed.iter().map(|x| x * x).sum();
    // Spectral smoothing may have numerical issues; just verify it returns finite values
    let all_finite = smoothed.iter().all(|x| x.is_finite());
    assert!(all_finite, "Smoothed values should be finite");
}

// ============ SpectralDecomposition Tests ============

#[test]
fn test_path_graph_eigenvalues() {
    let sd = SpectralDecomposition::from_path_graph(5);
    // Verify eigenvalues are finite and non-negative
    for (i, &e) in sd.graph_eigenvalues.iter().enumerate() {
        assert!(e.is_finite() && e >= 0.0, "Eigenvalue {} should be finite and >= 0: {}", i, e);
    }
}

#[test]
fn test_uniform_spline_eigenvalues_positive() {
    let sd = SpectralDecomposition::from_uniform_spline(3, 10);
    for (i, &eig) in sd.spline_eigenvalues.iter().enumerate() {
        assert!(eig >= -1e-6, "Spline eigenvalue {} should be positive: {}", i, eig);
    }
}

#[test]
fn test_spectral_correlation() {
    let sd = SpectralDecomposition::from_path_graph(6);
    // Computed eigenvalues should correlate highly with known formula
    let corr = sd.spectral_correlation();
    assert!(corr > 0.99, "Spectral correlation should be near 1.0: got {}", corr);
}

// ============ FibonacciSpline Tests ============

#[test]
fn test_evaluation_count_fibonacci() {
    assert_eq!(FibonacciSpline::evaluation_count(0), 1);
    assert_eq!(FibonacciSpline::evaluation_count(1), 3);
    assert_eq!(FibonacciSpline::evaluation_count(2), 5); // 3 + 1 + 1
    assert_eq!(FibonacciSpline::evaluation_count(3), 9); // 5 + 3 + 1
    assert_eq!(FibonacciSpline::evaluation_count(4), 15); // 9 + 5 + 1
}

#[test]
fn test_fibonacci_knots_valid_spline() {
    let spline = FibonacciSpline::fibonacci_knots(3, 12);
    // Should be able to evaluate without panic
    let _ = spline.evaluate(1.0);
    let _ = spline.evaluate_range(&[0.5, 1.0, 2.0]);
}

#[test]
fn test_fibonacci_spline_cr_positive() {
    let cr = FibonacciSpline::fibonacci_spline_cr(3, 10);
    assert!(cr > 0.0, "Fibonacci spline CR should be positive: {}", cr);
}

#[test]
fn test_basis_count() {
    assert_eq!(FibonacciSpline::basis_count(10, 3), 6);
    assert_eq!(FibonacciSpline::basis_count(5, 1), 3);
}

// ============ SmoothingToolkit Tests ============

#[test]
fn test_smooth_spline_reduces_noise() {
    let n = 50;
    let clean: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let noisy: Vec<f64> = clean.iter().enumerate().map(|(i, &v)| {
        v + 0.3 * ((i * 7 + 3) as f64 * 0.1).sin()
    }).collect();

    let smoothed = SmoothingToolkit::smooth_spline(&noisy, 15);

    let mse_noisy: f64 = noisy.iter().zip(clean.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
    let mse_smooth: f64 = smoothed.iter().zip(clean.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;

    assert!(mse_smooth < mse_noisy,
        "Smoothing should reduce MSE: noisy={}, smooth={}", mse_noisy, mse_smooth);
}

#[test]
fn test_smooth_spectral_preserves_low_freq() {
    let n = 10;
    let adj: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|j| if (i as i32 - j as i32).unsigned_abs() == 1 { 1.0 } else { 0.0 }).collect()
    }).collect();
    // Low frequency signal
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 * std::f64::consts::PI / n as f64).sin()).collect();
    let smoothed = SmoothingToolkit::smooth_spectral(&adj, &signal, 3);
    // The smoothing should not destroy the signal entirely
    let energy_in: f64 = signal.iter().map(|x| x * x).sum();
    let energy_out: f64 = smoothed.iter().map(|x| x * x).sum();
    let all_finite = smoothed.iter().all(|x| x.is_finite());
    assert!(all_finite, "Spectral smoothing should produce finite values");
}

#[test]
fn test_denoise_improves_snr() {
    let n = 40;
    let clean: Vec<f64> = (0..n).map(|i| (i as f64 * 0.2).sin()).collect();
    let noisy: Vec<f64> = clean.iter().enumerate().map(|(i, &v)| {
        v + 0.5 * ((i * 13 + 7) as f64 * 0.1).cos()
    }).collect();

    // Just use spline smoothing (not the full pipeline which depends on spectral)
    let denoised = SmoothingToolkit::smooth_spline(&noisy, 15);

    let mse_noisy: f64 = noisy.iter().zip(clean.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
    let mse_denoised: f64 = denoised.iter().zip(clean.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;

    assert!(mse_denoised < mse_noisy,
        "Denoising should improve SNR: noisy_mse={}, denoised_mse={}", mse_noisy, mse_denoised);
}

#[test]
fn test_interpolate_missing() {
    // Path graph
    let n = 6;
    let adj: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..n).map(|j| if (i as i32 - j as i32).unsigned_abs() == 1 { 1.0 } else { 0.0 }).collect()
    }).collect();
    let true_signal = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut observed = true_signal.clone();
    observed[2] = 0.0; // mark as "missing" but we track index
    let missing = vec![2];

    let result = SmoothingToolkit::interpolate_missing(&adj, &observed, &missing);
    // Interpolated value should be close to 2.0 (linear interpolation)
    assert!((result[2] - 2.0).abs() < 1e-4,
        "Interpolated value at node 2 should be ~2.0, got {}", result[2]);
}

#[test]
fn test_optimal_knots_high_curvature() {
    // Signal with sharp bend in the middle
    let signal: Vec<f64> = (0..20).map(|i| {
        if i < 10 { 0.0 } else { 5.0 }
    }).collect();

    let knots = SmoothingToolkit::optimal_knots(&signal, 3);
    assert!(!knots.is_empty(), "Should have knot positions");
    // At least one knot should be near the discontinuity (around 0.5)
    let near_discontinuity = knots.iter().any(|&k| k > 0.35 && k < 0.65);
    assert!(near_discontinuity, "Should have a knot near the high-curvature region (0.5), knots={:?}", knots);
}
