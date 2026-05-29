/// B-spline basis functions via Cox-de Boor recurrence
pub struct BSpline {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub control_points: Vec<f64>,
}

impl BSpline {
    pub fn new(degree: usize, knots: Vec<f64>, control_points: Vec<f64>) -> BSpline {
        assert!(knots.len() == control_points.len() + degree + 1,
            "knots.len() must equal control_points.len() + degree + 1");
        BSpline { degree, knots, control_points }
    }

    /// Cox-de Boor recurrence
    pub fn basis(&self, i: usize, p: usize, t: f64) -> f64 {
        if p == 0 {
            if self.knots[i] <= t && t < self.knots[i + 1] { 1.0 } else { 0.0 }
        } else {
            let denom_left = self.knots[i + p] - self.knots[i];
            let left = if denom_left.abs() > 1e-12 {
                (t - self.knots[i]) / denom_left * self.basis(i, p - 1, t)
            } else {
                0.0
            };
            let denom_right = self.knots[i + p + 1] - self.knots[i + 1];
            let right = if denom_right.abs() > 1e-12 {
                (self.knots[i + p + 1] - t) / denom_right * self.basis(i + 1, p - 1, t)
            } else {
                0.0
            };
            left + right
        }
    }

    pub fn evaluate(&self, t: f64) -> f64 {
        let n = self.control_points.len();
        let mut result = 0.0;
        for i in 0..n {
            result += self.control_points[i] * self.basis(i, self.degree, t);
        }
        result
    }

    pub fn evaluate_range(&self, ts: &[f64]) -> Vec<f64> {
        ts.iter().map(|&t| self.evaluate(t)).collect()
    }

    /// Collocation matrix B_{ij} = N_{j,p}(t_i)
    pub fn collocation_matrix(&self, sample_points: &[f64]) -> Vec<Vec<f64>> {
        let n = self.control_points.len();
        sample_points.iter().map(|&t| {
            (0..n).map(|j| self.basis(j, self.degree, t)).collect()
        }).collect()
    }
}
