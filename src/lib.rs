pub mod bspline;
pub mod graph_spline;
pub mod spectral;
pub mod fibonacci;
pub mod smoothing;

pub use bspline::BSpline;
pub use graph_spline::GraphSpline;
pub use spectral::SpectralDecomposition;
pub use fibonacci::FibonacciSpline;
pub use smoothing::SmoothingToolkit;

#[cfg(test)]
mod tests;
