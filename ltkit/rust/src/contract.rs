/// Contract types for LTKit backends.
///
/// The backend contract is expressed through six capabilities:
/// `parameter_groups`, `scores`, `apply_mask`, `snapshot`, `restore`, and `fit/evaluate`.
/// Implementations must preserve the canonical-order invariant for `scores` and `apply_mask`,
/// and the mask-persistence invariant across `fit` and `restore`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Criterion {
    Magnitude,
    Gate,
    Random,
    Snip,
}

/// Rewind strategy for `restore`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RewindPolicy {
    Init,
    EarlyK,
    None,
}

/// A prunable model backend.
///
/// This trait defines the six-portability contract used by LTKit. A conforming
/// implementation must support the six verbs:
/// `parameter_groups`, `scores`, `apply_mask`, `snapshot`, `restore`, and `fit/evaluate`.
///
/// Canonical-order invariant:
/// for any parameter group, the elements returned by `scores` must align
/// element-for-element with the model layout targeted by `apply_mask`.
///
/// Mask-persistence invariant:
/// once `apply_mask` zeros an element, that zero must persist through `fit`
/// and survive `restore`.
pub trait PrunableModel {
    type State;

    /// Enumerate the prunable parameter groups in stable canonical order.
    fn parameter_groups(&self) -> Vec<String>;

    /// Return per-element scores for one parameter group in canonical order.
    fn scores(&self, name: &str, criterion: Criterion) -> Vec<f32>;

    /// Apply a boolean mask in canonical order.
    fn apply_mask(&mut self, name: &str, mask: &[bool]);

    /// Capture an opaque snapshot for later `restore`.
    fn snapshot(&self) -> Self::State;

    /// Restore a previously captured snapshot without unmasking pruned elements.
    fn restore(&mut self, state: &Self::State);

    /// Train the masked model in place for `epochs`.
    fn fit(&mut self, epochs: usize);

    /// Evaluate the current model and return the metric value.
    fn evaluate(&self) -> f32;
}
