/// Splines on arbitrary graphs via spectral filtering
pub struct GraphSpline {
    pub adj: Vec<Vec<f64>>,
    pub constraints: Vec<(usize, f64)>,
}

impl GraphSpline {
    pub fn new(adj: Vec<Vec<f64>>) -> GraphSpline {
        GraphSpline { adj, constraints: Vec::new() }
    }

    pub fn constrain(&mut self, node: usize, value: f64) {
        self.constraints.push((node, value));
    }

    /// Compute graph Laplacian: L = D - A
    fn laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.adj.len();
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            let deg: f64 = self.adj[i].iter().sum();
            l[i][i] = deg;
            for j in 0..n {
                l[i][j] -= self.adj[i][j];
            }
        }
        l
    }

    /// Matrix-vector multiply
    fn mat_vec(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        let n = mat.len();
        (0..n).map(|i| {
            (0..n).map(|j| mat[i][j] * v[j]).sum()
        }).collect()
    }

    /// Solve for smoothest function satisfying constraints
    /// minimize f^T L f subject to f[i] = v[i] for constrained nodes
    /// Partition L into constrained (c) and free (f) blocks:
    /// L = [L_ff  L_fc; L_cf  L_cc]
    /// Optimal: L_ff * x_f = -L_fc * x_c
    pub fn interpolate(&self) -> Vec<f64> {
        let n = self.adj.len();
        let lap = self.laplacian();
        let constrained_set: Vec<bool> = {
            let mut s = vec![false; n];
            for &(idx, _) in &self.constraints {
                if idx < n { s[idx] = true; }
            }
            s
        };

        let free_indices: Vec<usize> = (0..n).filter(|&i| !constrained_set[i]).collect();
        let constrained_indices: Vec<usize> = (0..n).filter(|&i| constrained_set[i]).collect();

        if free_indices.is_empty() {
            let mut result = vec![0.0; n];
            for &(idx, val) in &self.constraints {
                if idx < n { result[idx] = val; }
            }
            return result;
        }

        // x_c vector
        let mut x_c = vec![0.0; constrained_indices.len()];
        for &(idx, val) in &self.constraints {
            if let Some(pos) = constrained_indices.iter().position(|&i| i == idx) {
                x_c[pos] = val;
            }
        }

        // Extract L_ff submatrix
        let nf = free_indices.len();
        let mut l_ff = vec![vec![0.0; nf]; nf];
        for (fi, &i) in free_indices.iter().enumerate() {
            for (fj, &j) in free_indices.iter().enumerate() {
                l_ff[fi][fj] = lap[i][j];
            }
        }

        // Extract L_fc submatrix
        let nc = constrained_indices.len();
        let mut l_fc = vec![vec![0.0; nc]; nf];
        for (fi, &i) in free_indices.iter().enumerate() {
            for (ci, &j) in constrained_indices.iter().enumerate() {
                l_fc[fi][ci] = lap[i][j];
            }
        }

        // rhs = -L_fc * x_c
        let mut rhs = vec![0.0; nf];
        for fi in 0..nf {
            for ci in 0..nc {
                rhs[fi] -= l_fc[fi][ci] * x_c[ci];
            }
        }

        // Solve L_ff * x_f = rhs via Gaussian elimination
        let x_f = Self::solve_linear(&l_ff, &rhs);

        // Assemble full result
        let mut result = vec![0.0; n];
        for (fi, &i) in free_indices.iter().enumerate() {
            result[i] = x_f[fi];
        }
        for &(idx, val) in &self.constraints {
            if idx < n { result[idx] = val; }
        }
        result
    }

    /// Gaussian elimination with partial pivoting
    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = a[i][j];
            }
            aug[i][n] = b[i];
        }

        for col in 0..n {
            // Partial pivot
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
            if pivot.abs() < 1e-12 { continue; }

            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-12 { continue; }
            x[i] = aug[i][n];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
        x
    }

    pub fn smoothness(&self, f: &[f64]) -> f64 {
        let lap = self.laplacian();
        let lf = Self::mat_vec(&lap, f);
        f.iter().zip(lf.iter()).map(|(&a, &b)| a * b).sum()
    }

    /// Compute eigenvalues of Laplacian using Jacobi iteration (for symmetric matrices)
    fn eigenvalues(mat: &[Vec<f64>], max_iter: usize) -> Vec<f64> {
        let n = mat.len();
        let mut a = mat.to_vec();
        for _ in 0..max_iter {
            // Find largest off-diagonal element
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

            // Jacobi rotation
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
        (0..n).map(|i| a[i][i]).collect()
    }

    /// Eigenvectors via power iteration (top k)
    fn eigenvectors_top_k(mat: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<Vec<f64>> {
        let n = mat.len();
        let mut eigenvecs: Vec<Vec<f64>> = Vec::new();
        let mut deflated = mat.to_vec();

        for _ in 0..k {
            let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in v.iter_mut() { *x /= norm; }

            for _ in 0..max_iter {
                let mut new_v = Self::mat_vec(&deflated, &v);
                // Orthogonalize against previous eigenvectors
                for ev in &eigenvecs {
                    let dot: f64 = new_v.iter().zip(ev.iter()).map(|(&a, &b)| a * b).sum();
                    for (j, &val) in ev.iter().enumerate() {
                        new_v[j] -= dot * val;
                    }
                }
                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-12 { break; }
                for x in v.iter_mut() { *x /= norm; }
                v = new_v;
                let norm2: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm2 < 1e-12 { break; }
                for x in v.iter_mut() { *x /= norm2; }
            }
            eigenvecs.push(v.clone());

            // Deflate
            let lambda: f64 = {
                let av = Self::mat_vec(&deflated, &v);
                v.iter().zip(av.iter()).map(|(&a, &b)| a * b).sum()
            };
            for i in 0..n {
                for j in 0..n {
                    deflated[i][j] -= lambda * v[i] * v[j];
                }
            }
        }
        eigenvecs
    }

    /// Low-pass spectral filter: keep only first k eigenvectors
    pub fn spectral_smooth(&self, signal: &[f64], k: usize) -> Vec<f64> {
        let n = self.adj.len();
        let lap = self.laplacian();

        // Get eigenvalues and eigenvectors
        let eigenvalues = Self::eigenvalues(&lap, 200);
        let mut indexed: Vec<(f64, usize)> = eigenvalues.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Get eigenvectors for smallest k eigenvalues
        let eigvecs = Self::eigenvectors_top_k(&lap, k.min(n), 300);

        // Project signal onto top k eigenvectors
        let mut result = vec![0.0; n];
        for v in &eigvecs {
            let coeff: f64 = signal.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
            for i in 0..n {
                result[i] += coeff * v[i];
            }
        }
        result
    }
}
