use crate::bspline::BSpline;
use crate::graph_spline::GraphSpline;

pub struct SmoothingToolkit;

impl SmoothingToolkit {
    /// Smooth signal using cubic B-spline
    pub fn smooth_spline(signal: &[f64], num_knots: usize) -> Vec<f64> {
        let n = signal.len();
        if n < 2 { return signal.to_vec(); }

        let degree = 3;
        let n_knots = num_knots.max(degree + 2);
        let n_control = n_knots - degree - 1;

        // Uniform clamped knot vector
        let mut knots = Vec::new();
        for _ in 0..=degree { knots.push(0.0); }
        for i in 1..n_knots - 2 * degree - 1 {
            knots.push(i as f64 / (n_knots - 2 * degree - 1) as f64);
        }
        for _ in 0..=degree { knots.push(1.0); }

        // Ensure correct knot count
        while knots.len() < n_control + degree + 1 { knots.push(1.0); }
        knots.truncate(n_control + degree + 1);

        // Least-squares fit: sample signal at uniform points
        let sample_ts: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let mut spline = BSpline::new(degree, knots, vec![0.0; n_control]);

        let colmat = spline.collocation_matrix(&sample_ts);

        // Solve normal equations: (B^T B) c = B^T y via Gaussian elimination
        let nc = n_control;
        let mut ata = vec![vec![0.0; nc]; nc];
        let mut aty = vec![0.0; nc];

        for i in 0..nc {
            for j in 0..nc {
                for k in 0..n {
                    ata[i][j] += colmat[k][i] * colmat[k][j];
                }
            }
            for k in 0..n {
                aty[i] += colmat[k][i] * signal[k];
            }
        }

        // Add small ridge for numerical stability
        for i in 0..nc {
            ata[i][i] += 1e-6;
        }

        let coeffs = Self::solve_linear(&ata, &aty);
        spline.control_points = coeffs;

        spline.evaluate_range(&sample_ts)
    }

    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n { aug[i][j] = a[i][j]; }
            aug[i][n] = b[i];
        }

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);
            let pivot = aug[col][col];
            if pivot.abs() < 1e-14 { continue; }
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-14 { continue; }
            x[i] = aug[i][n];
            for j in (i + 1)..n { x[i] -= aug[i][j] * x[j]; }
            x[i] /= aug[i][i];
        }
        x
    }

    /// Spectral smoothing on arbitrary graph
    pub fn smooth_spectral(adj: &[Vec<f64>], signal: &[f64], k: usize) -> Vec<f64> {
        let gs = GraphSpline::new(adj.to_vec());
        gs.spectral_smooth(signal, k)
    }

    /// Spline-spectral hybrid denoising
    pub fn denoise(signal: &[f64], degree: usize, cutoff: usize) -> Vec<f64> {
        let n = signal.len();
        if n < 4 { return signal.to_vec(); }

        // Step 1: Fit spline
        let num_knots = (n / 4).max(degree + 2);
        let smoothed = Self::smooth_spline(signal, num_knots);

        // Step 2: Build path graph Laplacian
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n - 1 {
            adj[i][i + 1] = 1.0;
            adj[i + 1][i] = 1.0;
        }

        // Step 3: Spectral filter the spline-smoothed signal
        Self::smooth_spectral(&adj, &smoothed, cutoff)
    }

    /// Interpolate missing values in a graph signal
    pub fn interpolate_missing(adj: &[Vec<f64>], signal: &[f64], missing: &[usize]) -> Vec<f64> {
        let mut gs = GraphSpline::new(adj.to_vec());
        for (i, &val) in signal.iter().enumerate() {
            if !missing.contains(&i) {
                gs.constrain(i, val);
            }
        }
        gs.interpolate()
    }

    /// Optimal knot positions (adaptive: place where curvature is highest)
    pub fn optimal_knots(signal: &[f64], num_knots: usize) -> Vec<f64> {
        let n = signal.len();
        if n < 3 || num_knots == 0 { return vec![]; }

        // Compute second differences (discrete curvature)
        let mut curvature = vec![0.0; n];
        for i in 1..n - 1 {
            curvature[i] = (signal[i - 1] - 2.0 * signal[i] + signal[i + 1]).abs();
        }
        curvature[0] = curvature[1];
        curvature[n - 1] = curvature[n - 2];

        // Place knots at highest curvature positions
        let mut indexed: Vec<(f64, usize)> = curvature.iter().enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        indexed.truncate(num_knots);

        let mut positions: Vec<f64> = indexed.iter().map(|&(_, i)| i as f64 / (n - 1) as f64).collect();
        positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        positions
    }
}
