use crate::bspline::BSpline;

pub struct FibonacciSpline {
    pub degree: usize,
    pub depths: Vec<usize>,
}

impl FibonacciSpline {
    /// evaluation_count follows: count(p) = count(p-1) + count(p-2) + 1
    pub fn evaluation_count(degree: usize) -> u64 {
        if degree == 0 { return 1; }
        if degree == 1 { return 3; } // 2 base evaluations + 1 combination
        let mut a = 1u64;
        let mut b = 3u64;
        for _ in 2..=degree {
            let c = b + a + 1;
            a = b;
            b = c;
        }
        b
    }

    /// Number of basis functions at given degree with n_knots
    pub fn basis_count(n_knots: usize, degree: usize) -> usize {
        if n_knots <= degree + 1 { return 0; }
        n_knots - degree - 1
    }

    pub fn fibonacci_knots(degree: usize, n_knots: usize) -> BSpline {
        let mut fib = vec![1u64, 1u64];
        while fib.len() < n_knots {
            let next = fib[fib.len() - 1] + fib[fib.len() - 2];
            fib.push(next);
        }

        let n_control = if n_knots > degree + 1 { n_knots - degree - 1 } else { 1 };
        let total_knots = n_control + degree + 1;

        // Build knot vector from Fibonacci sequence, clamped at ends
        let mut knots = Vec::new();
        for i in 0..total_knots {
            if i < degree + 1 {
                knots.push(0.0);
            } else if i >= total_knots - degree - 1 {
                knots.push(fib[n_knots.saturating_sub(1)] as f64);
            } else {
                let fi = (i - degree).min(fib.len() - 1);
                knots.push(fib[fi] as f64);
            }
        }

        // Ensure non-decreasing knots
        for i in 1..knots.len() {
            if knots[i] < knots[i - 1] {
                knots[i] = knots[i - 1];
            }
        }

        let control_points = vec![1.0; n_control];
        BSpline::new(degree, knots, control_points)
    }

    /// CR of Fibonacci-spaced spline collocation matrix
    pub fn fibonacci_spline_cr(degree: usize, n_knots: usize) -> f64 {
        let spline = Self::fibonacci_knots(degree, n_knots);
        let n = spline.control_points.len();
        if n < 2 { return 0.0; }

        let t_min = spline.knots[degree];
        let t_max = spline.knots[n + degree];
        let step = (t_max - t_min) / (n as f64);

        let sample_points: Vec<f64> = (0..n)
            .map(|i| t_min + (i as f64 + 0.5) * step)
            .collect();

        let colmat = spline.collocation_matrix(&sample_points);

        // B^T B
        let mut btb = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n.min(colmat.len()) {
                    btb[i][j] += colmat[k][i] * colmat[k][j];
                }
            }
        }

        // Trace and Frobenius norm
        let trace: f64 = (0..n).map(|i| btb[i][i]).sum();
        let frob_sq: f64 = btb.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum();

        if frob_sq < 1e-12 { return 0.0; }
        trace * trace / (n as f64 * frob_sq)
    }
}
