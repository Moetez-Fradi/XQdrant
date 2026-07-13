//! Masked-distance kernels: **repack** (scratch buffer + contiguous SIMD) vs **gather**
//! (score focus dims in place, optionally via `vgatherdps` on x86).
//!
//! Selectable at runtime via `XQDRANT_MASKED_KERNEL=repack|gather` (default: `repack`).

use std::sync::OnceLock;

use common::types::ScoreType;

use crate::types::Distance;

/// How to evaluate a masked (focus-dim) distance against a full stored vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MaskedKernelMode {
    /// Copy focus dims into a contiguous scratch buffer, then call the normal SIMD metric.
    #[default]
    Repack,
    /// Accumulate the metric by gathering focus dims from the stored vector (no scratch
    /// of the stored side). On x86_64 with AVX2 this uses `_mm256_i32gather_ps`.
    Gather,
}

impl MaskedKernelMode {
    /// Resolve from `XQDRANT_MASKED_KERNEL` (`repack` / `gather`). Unknown/missing → Repack.
    pub fn from_env() -> Self {
        static CACHED: OnceLock<MaskedKernelMode> = OnceLock::new();
        *CACHED.get_or_init(|| match std::env::var("XQDRANT_MASKED_KERNEL") {
            Ok(v) if v.eq_ignore_ascii_case("gather") => Self::Gather,
            _ => Self::Repack,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Repack => "repack",
            Self::Gather => "gather",
        }
    }
}

/// Repack focus dims of `vector` into `scratch` (must be `dims.len()`), then score with
/// the provided contiguous similarity function (typically `TMetric::similarity`).
#[inline]
pub fn score_repack_f32(
    query: &[f32],
    vector: &[f32],
    dims: &[usize],
    scratch: &mut [f32],
    similarity: fn(&[f32], &[f32]) -> ScoreType,
) -> ScoreType {
    debug_assert_eq!(query.len(), dims.len());
    debug_assert_eq!(scratch.len(), dims.len());
    for (slot, &dim) in scratch.iter_mut().zip(dims.iter()) {
        *slot = vector[dim];
    }
    similarity(query, scratch)
}

/// Gather-based masked score for f32 (Dot / Cosine / Euclid / Manhattan).
#[inline]
pub fn score_gather_f32(
    distance: Distance,
    query: &[f32],
    vector: &[f32],
    dims: &[usize],
) -> ScoreType {
    debug_assert_eq!(query.len(), dims.len());
    match distance {
        Distance::Dot | Distance::Cosine => gather_dot(query, vector, dims),
        Distance::Euclid => gather_euclid(query, vector, dims),
        Distance::Manhattan => gather_manhattan(query, vector, dims),
    }
}

#[inline]
fn gather_dot(query: &[f32], vector: &[f32], dims: &[usize]) -> ScoreType {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && query.len() >= 8 {
            return unsafe { gather_dot_avx2(query, vector, dims) };
        }
    }
    gather_dot_scalar(query, vector, dims)
}

#[inline]
fn gather_dot_scalar(query: &[f32], vector: &[f32], dims: &[usize]) -> ScoreType {
    let mut sum = 0.0;
    for (i, &dim) in dims.iter().enumerate() {
        sum += query[i] * vector[dim];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gather_dot_avx2(query: &[f32], vector: &[f32], dims: &[usize]) -> ScoreType {
    use std::arch::x86_64::*;

    unsafe {
        let n = query.len();
        let m = n - (n % 8);
        let mut sum = _mm256_setzero_ps();
        let base = vector.as_ptr();

        let mut i = 0;
        while i < m {
            // Element indices as i32 for vgatherdps (scale = 4 → byte offset = idx * 4).
            let idx = _mm256_set_epi32(
                dims[i + 7] as i32,
                dims[i + 6] as i32,
                dims[i + 5] as i32,
                dims[i + 4] as i32,
                dims[i + 3] as i32,
                dims[i + 2] as i32,
                dims[i + 1] as i32,
                dims[i] as i32,
            );
            let gathered = _mm256_i32gather_ps(base, idx, 4);
            let q = _mm256_loadu_ps(query.as_ptr().add(i));
            sum = _mm256_fmadd_ps(q, gathered, sum);
            i += 8;
        }

        // Horizontal sum of AVX register.
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let mut result = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

        while i < n {
            result += query[i] * vector[dims[i]];
            i += 1;
        }
        result
    }
}

#[inline]
fn gather_euclid(query: &[f32], vector: &[f32], dims: &[usize]) -> ScoreType {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && query.len() >= 8 {
            return unsafe { gather_euclid_avx2(query, vector, dims) };
        }
    }
    let mut sum = 0.0;
    for (i, &dim) in dims.iter().enumerate() {
        let d = query[i] - vector[dim];
        sum += d * d;
    }
    -sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gather_euclid_avx2(query: &[f32], vector: &[f32], dims: &[usize]) -> ScoreType {
    use std::arch::x86_64::*;

    unsafe {
        let n = query.len();
        let m = n - (n % 8);
        let mut sum = _mm256_setzero_ps();
        let base = vector.as_ptr();

        let mut i = 0;
        while i < m {
            let idx = _mm256_set_epi32(
                dims[i + 7] as i32,
                dims[i + 6] as i32,
                dims[i + 5] as i32,
                dims[i + 4] as i32,
                dims[i + 3] as i32,
                dims[i + 2] as i32,
                dims[i + 1] as i32,
                dims[i] as i32,
            );
            let gathered = _mm256_i32gather_ps(base, idx, 4);
            let q = _mm256_loadu_ps(query.as_ptr().add(i));
            let diff = _mm256_sub_ps(q, gathered);
            sum = _mm256_fmadd_ps(diff, diff, sum);
            i += 8;
        }

        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let mut result = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

        while i < n {
            let d = query[i] - vector[dims[i]];
            result += d * d;
            i += 1;
        }
        -result
    }
}

#[inline]
fn gather_manhattan(query: &[f32], vector: &[f32], dims: &[usize]) -> ScoreType {
    let mut sum = 0.0;
    for (i, &dim) in dims.iter().enumerate() {
        sum += (query[i] - vector[dim]).abs();
    }
    -sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::simple::{dot_similarity, euclid_similarity, manhattan_similarity};

    #[test]
    fn gather_matches_repack_dot() {
        let query = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let vector: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let dims: Vec<usize> = (0..9).map(|i| i * 3 + 1).collect();
        let mut scratch = vec![0.0; dims.len()];

        let repack = score_repack_f32(&query, &vector, &dims, &mut scratch, dot_similarity);
        let gather = score_gather_f32(Distance::Dot, &query, &vector, &dims);
        assert!(
            (repack - gather).abs() < 1e-4,
            "repack={repack} gather={gather}"
        );
    }

    #[test]
    fn gather_matches_repack_euclid_manhattan() {
        let query = [0.5f32, -1.0, 2.0, 0.0, 3.0, -0.5, 1.5, 2.5];
        let vector: Vec<f32> = (0..32).map(|i| (i as f32 - 10.0) * 0.25).collect();
        let dims: Vec<usize> = vec![0, 2, 4, 7, 11, 13, 20, 31];
        let mut scratch = vec![0.0; dims.len()];

        for (distance, sim) in [
            (
                Distance::Euclid,
                euclid_similarity as fn(&[f32], &[f32]) -> ScoreType,
            ),
            (
                Distance::Manhattan,
                manhattan_similarity as fn(&[f32], &[f32]) -> ScoreType,
            ),
        ] {
            let repack = score_repack_f32(&query, &vector, &dims, &mut scratch, sim);
            let gather = score_gather_f32(distance, &query, &vector, &dims);
            assert!(
                (repack - gather).abs() < 1e-4,
                "{distance:?}: repack={repack} gather={gather}"
            );
        }
    }

    #[test]
    fn kernel_mode_as_str() {
        assert_eq!(MaskedKernelMode::Repack.as_str(), "repack");
        assert_eq!(MaskedKernelMode::Gather.as_str(), "gather");
    }
}
