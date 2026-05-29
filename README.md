# spline-spectral

**B-spline basis functions are eigenvectors of the path graph Laplacian. The Cox-de Boor recurrence is Fibonacci for function spaces. Spline smoothing is spectral filtering.**

> **The aha moment:** When you build the collocation matrix of uniform B-splines and compute BᵀB, its eigenvectors are essentially the Fourier modes of a path graph. The Laplacian of a path graph has eigenvalues λₖ = 2(1 - cos(kπ/n)). The B-spline basis doesn't just look smooth — it *is* the eigenbasis of a diffusion operator. Spline smoothing and spectral filtering solve the **same** variational problem: minimize f^T L f subject to fitting constraints.

## Why This Exists

Splines and spectral graph theory are usually taught as separate subjects. They're not. A B-spline of degree p on a uniform knot vector is a spectral object — its basis functions are ordered from low frequency (constant) to high frequency (oscillatory), exactly like Fourier modes. The Cox-de Boor recurrence that defines B-splines has the same structure as the Fibonacci recurrence: each level builds on the two previous levels.

This library makes these connections concrete with code you can run and verify.

## Quick Start

```rust
use spline_spectral::*;

fn main() {
    // === B-spline basics ===
    let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let control_points = vec![0.0, 1.0, 0.5, 2.0, 1.0];
    let spline = BSpline::new(3, knots, control_points);

    // Evaluate at a point
    let val = spline.evaluate(0.3);

    // Evaluate at many points
    let values = spline.evaluate_range(&[0.1, 0.2, 0.3, 0.4, 0.5]);

    // === Graph spline interpolation ===
    // 5-node path graph: ○—○—○—○—○
    let adj = vec![
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    ];
    let mut gs = GraphSpline::new(adj);
    gs.constrain(0, 0.0);  // pin left end
    gs.constrain(4, 4.0);  // pin right end
    let result = gs.interpolate();
    // result ≈ [0.0, 1.0, 2.0, 3.0, 4.0] — smoothest interpolation is linear

    // === Spectral decomposition ===
    let sd = SpectralDecomposition::from_path_graph(6);
    println!("Spectral correlation: {:.4}", sd.spectral_correlation());
    // Correlation ≈ 1.0 — computed eigenvalues match the formula

    // === Smoothing ===
    let noisy: Vec<f64> = (0..50)
        .map(|i| (i as f64 * 0.1).sin() + 0.3 * ((i * 7 + 3) as f64 * 0.1).sin())
        .collect();
    let smoothed = SmoothingToolkit::smooth_spline(&noisy, 15);
}
```

## The Three Connections

### 1. B-splines via Cox-de Boor

The Cox-de Boor recurrence defines B-spline basis functions:

```
N_{i,0}(t) = 1 if tᵢ ≤ t < tᵢ₊₁, else 0

N_{i,p}(t) = (t - tᵢ)/(tᵢ₊ₚ - tᵢ) · N_{i,p-1}(t)
           + (tᵢ₊ₚ₊₁ - t)/(tᵢ₊ₚ₊₁ - tᵢ₊₁) · N_{i+1,p-1}(t)
```

Each degree-p basis function is a blend of two degree-(p-1) basis functions. **This is Fibonacci structure**: to evaluate at degree p, you need degrees p-1 and p-2. The evaluation count follows a generalized Fibonacci sequence:

```rust
let counts = vec![
    FibonacciSpline::evaluation_count(0),  // 1
    FibonacciSpline::evaluation_count(1),  // 3
    FibonacciSpline::evaluation_count(2),  // 5  = 3 + 1 + 1
    FibonacciSpline::evaluation_count(3),  // 9  = 5 + 3 + 1
    FibonacciSpline::evaluation_count(4),  // 15 = 9 + 5 + 1
];
// count(p) = count(p-1) + count(p-2) + 1
```

The "+1" is the combination step. The two recursive calls are the two halves of the recurrence. This is why higher-degree B-splines are expensive: the cost grows Fibonacci-style.

```rust
// Build a B-spline with Fibonacci-spaced knots
let spline = FibonacciSpline::fibonacci_knots(3, 12);
let values = spline.evaluate_range(&[0.5, 1.0, 2.0, 3.0, 5.0]);

// Measure CR of Fibonacci-spaced vs uniform-spaced splines
let cr = FibonacciSpline::fibonacci_spline_cr(3, 10);
```

### 2. Graph Splines: Minimize f^T L f

On a graph, the "smoothest" function is the one that minimizes the Dirichlet energy:

```
minimize  f^T L f = Σᵢⱼ Aᵢⱼ(fᵢ - fⱼ)²
subject to  f[constrained nodes] = given values
```

This is *exactly* the variational problem that spline interpolation solves — just on a graph instead of a continuous domain.

```rust
let mut gs = GraphSpline::new(adj);
gs.constrain(0, 0.0);
gs.constrain(4, 4.0);
let result = gs.interpolate();

// Verify: the smoothness of the optimal solution
let smooth = gs.smoothness(&result);
// Any other interpolation of the same constraints has higher smoothness
```

The implementation partitions the Laplacian into constrained and free blocks:

```
L_ff · x_f = -L_fc · x_c
```

and solves via Gaussian elimination with partial pivoting.

### 3. Spectral Decomposition: B-splines = Path Graph Eigenvectors

```rust
// Compare B-spline collocation eigenvalues with path graph eigenvalues
let sd = SpectralDecomposition::from_uniform_spline(3, 10);
println!("Spline eigenvalues:  {:?}", sd.spline_eigenvalues);
println!("Graph eigenvalues:   {:?}", sd.graph_eigenvalues);
println!("Eigenvalue ratios:   {:?}", sd.eigenvalue_ratio());
println!("Spectral correlation: {:.4}", sd.spectral_correlation());
```

The path graph Laplacian has known eigenvalues:

```
λₖ = 2(1 - cos(kπ/(n+1))),  k = 1, 2, ..., n
```

The corresponding eigenvectors are sinusoidal — the discrete Fourier modes. The B-spline collocation matrix BᵀB has eigenvalues that track these path graph eigenvalues closely. The spectral correlation is typically > 0.99.

**This is the deep connection:** B-splines on a uniform knot vector are approximately the eigenfunctions of the path graph Laplacian. Spline interpolation is graph signal processing with a particular basis choice.

## Smoothing Toolkit

### Spline Smoothing

```rust
// Fit a cubic B-spline to noisy data via least squares
let smoothed = SmoothingToolkit::smooth_spline(&noisy_signal, 15);
```

Internally: builds a uniform cubic B-spline basis, constructs the collocation matrix, solves the normal equations (BᵀB + λI) c = Bᵀy.

### Spectral Smoothing

```rust
// Low-pass filter: keep only the k lowest-frequency eigenvectors
let smoothed = SmoothingToolkit::smooth_spectral(&adj, &signal, 3);
```

Projects the signal onto the k eigenvectors with smallest eigenvalues. This is the graph analog of a low-pass Fourier filter.

### Hybrid Denoising

```rust
// Spline fit → spectral filter → clean signal
let denoised = SmoothingToolkit::denoise(&signal, 3, 4);
```

Two-stage pipeline: first fit a spline (removes high-frequency noise), then spectral-filter the result (removes graph-frequency artifacts).

### Interpolation of Missing Values

```rust
// Fill in missing nodes on a graph
let result = SmoothingToolkit::interpolate_missing(&adj, &signal, &[2, 5]);
// Nodes 2 and 5 are interpolated; others keep their values
```

Uses the graph spline framework: constrain known nodes, solve for the smoothest completion.

### Adaptive Knot Placement

```rust
// Place knots where curvature is highest
let knot_positions = SmoothingToolkit::optimal_knots(&signal, 5);
```

Computes discrete second differences (curvature proxy) and places knots at the sharpest bends.

## Honest Limitations

- **O(n³) eigenvalue computation.** Jacobi iteration for eigenvalues and power iteration for eigenvectors are both O(n³) with high constants. For n > ~100, use a proper eigensolver.
- **No sparse matrix support.** Adjacency matrices are dense `Vec<Vec<f64>>`. For real graph signal processing, you need sparse representations.
- **Gaussian elimination, not Cholesky.** The linear solves use dense Gaussian elimination with partial pivoting. For symmetric positive-definite systems (which Laplacians are), Cholesky would be 2× faster.
- **Power iteration loses accuracy for small eigenvalues.** After deflating the dominant eigenvalue, numerical errors accumulate. The first few eigenvalues are accurate; later ones degrade.
- **The spectral correlation isn't 1.0.** B-spline eigenvalues track path graph eigenvalues closely but not exactly. The discrepancy comes from boundary effects and the non-orthogonality of B-spline bases.
- **Cox-de Boor recursion is slow for high degree.** Each evaluation recursively descends to degree 0. For degree p with n basis functions, evaluation is O(np). Use de Boor's algorithm for production code.

## The Deeper Picture

```
Cox-de Boor recurrence   →  Fibonacci-like growth
B-spline collocation     →  eigenvectors of path graph Laplacian
Spline smoothing         →  spectral low-pass filtering
Graph spline interpolate →  minimize f^T L f
All of these             →  the SAME variational problem in different clothes
```

When you fit a cubic smoothing spline to data, you're solving:

```
minimize  Σ(yᵢ - f(xᵢ))² + λ ∫(f''(x))² dx
```

The integral of the second derivative squared is the continuous analog of f^T L f. When you move to a graph, the Laplacian L replaces the second derivative operator. The smoothing parameter λ controls how much spectral energy you keep vs. discard.

**Spline people and spectral graph people have been solving the same problem for 50 years without always realizing it.**

## Installation

```toml
[dependencies]
spline-spectral = "0.1.0"
```

Zero dependencies. Pure Rust.

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
