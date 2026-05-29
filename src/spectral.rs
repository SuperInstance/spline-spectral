use crate::bspline::BSpline;

pub struct SpectralDecomposition {
    pub spline_eigenvalues: Vec<f64>,
    pub graph_eigenvalues: Vec<f64>,
}

impl SpectralDecomposition {
    /// Compute eigenvalues via Jacobi iteration
    fn jacobi_eigenvalues(mat: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
        let n = mat.len();
        let mut a = mat.to_vec();
        for _ in 0..max_iter {
            let mut max_val = 0.0;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-12 { break; }

            let theta = if (a[p][p] - a[q][q]).abs() < 1e-12 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };
            let c = theta.cos();
            let s = theta.sin();

            for i in 0..n {
                if i != p && i != q {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip + s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = -s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];
            a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }
        let mut eigs: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eigs
    }

    /// Build uniform B-spline collocation matrix and compute its eigenvalues
    pub fn from_uniform_spline(degree: usize, n_knots: usize) -> SpectralDecomposition {
        let n = n_knots - degree - 1;
        // Uniform knots on [0, 1]
        let knots: Vec<f64> = (0..=n + degree).map(|i| i as f64).collect();
        let control_points = vec![0.0; n];

        let spline = BSpline::new(degree, knots, control_points);

        // Sample points at midpoints of interior knot spans
        let sample_points: Vec<f64> = (degree..degree + n)
            .map(|i| (spline.knots[i] + spline.knots[i + 1]) / 2.0)
            .collect();

        let colmat = spline.collocation_matrix(&sample_points);

        // Make it symmetric: use B^T B (which has meaningful eigenvalues)
        let n_mat = colmat.len();
        let mut btb = vec![vec![0.0; n_mat]; n_mat];
        for i in 0..n_mat {
            for j in 0..n_mat {
                for k in 0..n_mat {
                    btb[i][j] += colmat[k][i] * colmat[k][j];
                }
            }
        }

        let spline_eigs = Self::jacobi_eigenvalues(&btb, 500);

        // Corresponding path graph eigenvalues
        let graph_eigs = Self::path_graph_eigenvalues(n);

        SpectralDecomposition {
            spline_eigenvalues: spline_eigs,
            graph_eigenvalues: graph_eigs,
        }
    }

    /// Path graph Laplacian eigenvalues: λ_k = 2(1 - cos(kπ/n))
    fn path_graph_eigenvalues(n: usize) -> Vec<f64> {
        (1..=n).map(|k| 2.0 * (1.0 - (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos()))
            .collect()
    }

    pub fn from_path_graph(n: usize) -> SpectralDecomposition {
        // Build path graph Laplacian
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            if i > 0 {
                lap[i][i - 1] = -1.0;
                lap[i][i] += 1.0;
            }
            if i < n - 1 {
                lap[i][i + 1] = -1.0;
                lap[i][i] += 1.0;
            }
        }

        let computed_eigs = Self::jacobi_eigenvalues(&lap, 500);
        let known_eigs = Self::path_graph_eigenvalues(n);

        SpectralDecomposition {
            spline_eigenvalues: computed_eigs,
            graph_eigenvalues: known_eigs,
        }
    }

    pub fn eigenvalue_ratio(&self) -> Vec<f64> {
        let len = self.spline_eigenvalues.len().min(self.graph_eigenvalues.len());
        (0..len).map(|i| {
            if self.graph_eigenvalues[i].abs() > 1e-12 {
                self.spline_eigenvalues[i] / self.graph_eigenvalues[i]
            } else {
                0.0
            }
        }).collect()
    }

    pub fn spectral_correlation(&self) -> f64 {
        let len = self.spline_eigenvalues.len().min(self.graph_eigenvalues.len());
        if len == 0 { return 0.0; }

        let sx: Vec<f64> = self.spline_eigenvalues.iter().take(len).cloned().collect();
        let sy: Vec<f64> = self.graph_eigenvalues.iter().take(len).cloned().collect();

        let mean_x: f64 = sx.iter().sum::<f64>() / len as f64;
        let mean_y: f64 = sy.iter().sum::<f64>() / len as f64;

        let cov: f64 = sx.iter().zip(sy.iter()).map(|(&a, &b)| (a - mean_x) * (b - mean_y)).sum();
        let var_x: f64 = sx.iter().map(|a| (a - mean_x).powi(2)).sum();
        let var_y: f64 = sy.iter().map(|a| (a - mean_y).powi(2)).sum();

        if var_x < 1e-12 || var_y < 1e-12 { return 0.0; }
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}
