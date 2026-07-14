use std::borrow::Cow;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};

use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::Random;
use common::typelevel::False;
use common::types::{PointOffsetType, ScoreType};

use super::masked_kernels::{self, MaskedKernelMode};
use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::{TypedDenseVector, VectorElementType};
use crate::spaces::metric::Metric;
use crate::vector_storage::DenseVectorStorageRead;
use crate::vector_storage::query_scorer::QueryScorer;

/// Query scorer that computes the similarity of a dense query vector against stored
/// vectors using only a subset ("focus") of the vector dimensions.
///
/// Unlike the collection-level focus rescoring (preselect + rescore), this scorer plugs
/// directly into the index hot path: the masked distance is what drives HNSW traversal.
/// This is a real speedup on the per-comparison cost (fewer multiplies/adds), at the cost
/// of approximating the neighborhood — the graph itself was built for full vectors.
///
/// The query is preprocessed exactly like a regular nearest search (e.g. L2-normalized for
/// Cosine over *all* dimensions), and then the score sums only the terms of the focused
/// dimensions. This keeps the scores consistent with the collection-level focus rescoring.
///
/// Kernel selection (X1): [`MaskedKernelMode`] — `repack` (default) copies focus dims into
/// a scratch buffer then calls contiguous SIMD; `gather` scores in place (AVX2 `vgatherdps`
/// on x86). Override with `XQDRANT_MASKED_KERNEL=gather|repack`.
pub struct MaskedMetricQueryScorer<
    'a,
    TElement: PrimitiveVectorElement,
    TMetric: Metric<TElement>,
    TVectorStorage: DenseVectorStorageRead<TElement>,
> {
    vector_storage: &'a TVectorStorage,
    /// Preprocessed query, gathered down to the focus dimensions (in `dims` order).
    query: TypedDenseVector<TElement>,
    /// Focus dimension indexes, filtered to be within the vector bounds.
    dims: Vec<usize>,
    /// Reusable scratch buffer holding the focus dimensions of the vector being scored.
    /// Used by the repack kernel; unused by gather (except non-f32 fallback).
    buffer: RefCell<TypedDenseVector<TElement>>,
    kernel: MaskedKernelMode,
    metric: PhantomData<TMetric>,
    hardware_counter: HardwareCounterCell,
}

impl<
    'a,
    TElement: PrimitiveVectorElement,
    TMetric: Metric<TElement>,
    TVectorStorage: DenseVectorStorageRead<TElement>,
> MaskedMetricQueryScorer<'a, TElement, TMetric, TVectorStorage>
{
    pub fn new(
        query: TypedDenseVector<VectorElementType>,
        dims: &[u32],
        vector_storage: &'a TVectorStorage,
        hardware_counter: HardwareCounterCell,
    ) -> Self {
        Self::new_with_kernel(
            query,
            dims,
            vector_storage,
            hardware_counter,
            MaskedKernelMode::from_env(),
        )
    }

    /// Like [`Self::new`] but with an explicit kernel mode (used by tests / criterion).
    pub fn new_with_kernel(
        query: TypedDenseVector<VectorElementType>,
        dims: &[u32],
        vector_storage: &'a TVectorStorage,
        mut hardware_counter: HardwareCounterCell,
        kernel: MaskedKernelMode,
    ) -> Self {
        let vector_dim = vector_storage.vector_dim();

        // Preprocess the full query first (normalizes over all dimensions for Cosine),
        // then keep only the focus dimensions.
        let preprocessed_vector = TMetric::preprocess(query);
        let preprocessed: TypedDenseVector<TElement> =
            TElement::slice_from_float_cow(Cow::from(preprocessed_vector)).into_owned();

        // Keep only in-bounds dimensions, preserving the requested order. Out-of-bounds
        // dimensions are silently ignored, consistent with the focus rescoring path.
        let dims: Vec<usize> = dims
            .iter()
            .map(|&dim| dim as usize)
            .filter(|&dim| dim < vector_dim && dim < preprocessed.len())
            .collect();

        let query: TypedDenseVector<TElement> = dims.iter().map(|&dim| preprocessed[dim]).collect();

        let masked_len = dims.len();
        hardware_counter.set_cpu_multiplier(masked_len * size_of::<TElement>());
        // Even though we only score a subset of dimensions, the stored vector is read whole
        // from (potentially on-disk) storage, so I/O cost tracks the full vector width.
        if vector_storage.is_on_disk() {
            hardware_counter.set_vector_io_read_multiplier(vector_dim * size_of::<TElement>());
        } else {
            hardware_counter.set_vector_io_read_multiplier(0);
        }

        let buffer = RefCell::new(vec![TElement::default(); masked_len]);

        Self {
            vector_storage,
            query,
            dims,
            buffer,
            kernel,
            metric: PhantomData,
            hardware_counter,
        }
    }

    /// Score `vector` against the masked query using the configured kernel.
    #[inline]
    fn score_masked(&self, vector: &[TElement]) -> ScoreType {
        match self.kernel {
            MaskedKernelMode::Repack => self.score_repack(vector),
            MaskedKernelMode::Gather => self.score_gather(vector),
        }
    }

    #[inline]
    fn score_repack(&self, vector: &[TElement]) -> ScoreType {
        let mut buffer = self.buffer.borrow_mut();
        for (slot, &dim) in buffer.iter_mut().zip(self.dims.iter()) {
            *slot = vector[dim];
        }
        TMetric::similarity(&self.query, &buffer)
    }

    #[inline]
    fn score_gather(&self, vector: &[TElement]) -> ScoreType {
        // Fast path: f32 gather kernels (Dot/Cosine/Euclid/Manhattan).
        if size_of::<TElement>() == size_of::<f32>() && align_of::<TElement>() == align_of::<f32>()
        {
            let query_f32: &[f32] = bytemuck::cast_slice(&self.query);
            let vector_f32: &[f32] = bytemuck::cast_slice(vector);
            return masked_kernels::score_gather_f32(
                TMetric::distance(),
                query_f32,
                vector_f32,
                &self.dims,
            );
        }
        // Non-f32: fall back to repack (contiguous metric still wins for half/byte).
        self.score_repack(vector)
    }
}

impl<
    TElement: PrimitiveVectorElement,
    TMetric: Metric<TElement>,
    TVectorStorage: DenseVectorStorageRead<TElement>,
> QueryScorer for MaskedMetricQueryScorer<'_, TElement, TMetric, TVectorStorage>
{
    #[inline]
    fn score_stored(&self, idx: PointOffsetType) -> ScoreType {
        self.hardware_counter.cpu_counter().incr();
        self.hardware_counter.vector_io_read().incr();
        self.score_masked(&self.vector_storage.get_dense::<Random>(idx))
    }

    #[inline]
    fn score_stored_batch(&self, ids: &[PointOffsetType], scores: &mut [ScoreType]) {
        debug_assert_eq!(ids.len(), scores.len());

        self.hardware_counter.cpu_counter().incr_delta(ids.len());
        self.hardware_counter.vector_io_read().incr_delta(ids.len());

        self.vector_storage
            .for_each_in_dense_batch(ids, |idx, vector| {
                scores[idx] = self.score_masked(vector);
            })
            .expect("read vectors");
    }

    fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        self.hardware_counter.cpu_counter().incr();
        let v1 = self.vector_storage.get_dense::<Random>(point_a);
        let v2 = self.vector_storage.get_dense::<Random>(point_b);

        // Build a temporary masked view of point_a, then score point_b with the active kernel.
        let masked_a: TypedDenseVector<TElement> = self.dims.iter().map(|&dim| v1[dim]).collect();
        match self.kernel {
            MaskedKernelMode::Gather
                if size_of::<TElement>() == size_of::<f32>()
                    && align_of::<TElement>() == align_of::<f32>() =>
            {
                let query_f32: &[f32] = bytemuck::cast_slice(&masked_a);
                let vector_f32: &[f32] = bytemuck::cast_slice(v2.as_ref());
                masked_kernels::score_gather_f32(
                    TMetric::distance(),
                    query_f32,
                    vector_f32,
                    &self.dims,
                )
            }
            _ => {
                let mut buffer = self.buffer.borrow_mut();
                for (slot, &dim) in buffer.iter_mut().zip(self.dims.iter()) {
                    *slot = v2[dim];
                }
                TMetric::similarity(&masked_a, &buffer)
            }
        }
    }

    // Scoring raw bytes (the inline-vectors graph fast path) is not supported: that path
    // only kicks in for quantized storage, which masked focus search does not support.
    type SupportsBytes = False;
    fn score_bytes(&self, enabled: Self::SupportsBytes, _bytes: &[u8]) -> ScoreType {
        match enabled {}
    }
}

#[cfg(test)]
mod tests {
    use common::counter::hardware_counter::HardwareCounterCell;
    use common::types::PointOffsetType;

    use crate::data_types::vectors::{QueryVector, VectorInternal};
    use crate::types::Distance;
    use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
    use crate::vector_storage::{VectorStorage, new_raw_scorer};

    fn build_storage(
        distance: Distance,
        points: &[Vec<f32>],
    ) -> crate::vector_storage::VectorStorageEnum {
        let hw = HardwareCounterCell::new();
        let dim = points[0].len();
        let mut storage = new_volatile_dense_vector_storage(dim, distance);
        for (i, vec) in points.iter().enumerate() {
            storage
                .insert_vector(i as PointOffsetType, vec.as_slice().into(), &hw)
                .unwrap();
        }
        storage
    }

    #[test]
    fn masked_dot_scores_only_focus_dims() {
        let points = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
            vec![0.0, 1.0, 0.0, 1.0],
        ];
        let storage = build_storage(Distance::Dot, &points);

        let query = vec![1.0, 10.0, 100.0, 1000.0];
        let dims = vec![0u32, 2u32];
        let masked = QueryVector::NearestMasked(
            VectorInternal::Dense(query.clone()),
            dims.clone(),
            None,
            false,
            None,
        );
        let scorer = new_raw_scorer(masked, &storage, HardwareCounterCell::new()).unwrap();

        for (i, v) in points.iter().enumerate() {
            let expected: f32 = dims
                .iter()
                .map(|&d| query[d as usize] * v[d as usize])
                .sum();
            let got = scorer.score_point(i as PointOffsetType);
            assert!(
                (got - expected).abs() < 1e-4,
                "point {i}: got {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn masked_all_dims_equals_full_nearest() {
        let points = vec![
            vec![0.2, 0.5, 0.1, 0.9],
            vec![0.7, 0.1, 0.4, 0.3],
            vec![0.0, 0.8, 0.6, 0.2],
        ];

        for distance in [Distance::Dot, Distance::Cosine, Distance::Euclid] {
            let storage = build_storage(distance, &points);
            let query = vec![0.3, 0.4, 0.5, 0.6];
            let all_dims = vec![0u32, 1, 2, 3];

            let full = new_raw_scorer(
                QueryVector::Nearest(VectorInternal::Dense(query.clone())),
                &storage,
                HardwareCounterCell::new(),
            )
            .unwrap();
            let masked = new_raw_scorer(
                QueryVector::NearestMasked(
                    VectorInternal::Dense(query.clone()),
                    all_dims,
                    None,
                    false,
                    None,
                ),
                &storage,
                HardwareCounterCell::new(),
            )
            .unwrap();

            for i in 0..points.len() as PointOffsetType {
                let full_score = full.score_point(i);
                let masked_score = masked.score_point(i);
                assert!(
                    (full_score - masked_score).abs() < 1e-4,
                    "{distance:?} point {i}: full {full_score}, masked {masked_score}",
                );
            }
        }
    }

    #[test]
    fn masked_ignores_out_of_bounds_dims() {
        let points = vec![vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let storage = build_storage(Distance::Dot, &points);

        let query = vec![1.0, 1.0, 1.0];
        // dims 1 is valid, 5 and 9 are out of bounds and must be ignored.
        let masked = QueryVector::NearestMasked(
            VectorInternal::Dense(query.clone()),
            vec![1u32, 5u32, 9u32],
            None,
            false,
            None,
        );
        let scorer = new_raw_scorer(masked, &storage, HardwareCounterCell::new()).unwrap();

        assert!((scorer.score_point(0) - 2.0).abs() < 1e-4);
        assert!((scorer.score_point(1) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn hybrid_scorer_switches_by_layer() {
        use common::bitvec::BitVec;

        use crate::index::hnsw_index::point_scorer::FilteredScorer;

        let points = vec![
            vec![1.0, 0.0, 100.0],
            vec![0.0, 1.0, 100.0],
            vec![1.0, 1.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        // Focus dims [0, 1] ignore the large third component.
        let dims = vec![0u32, 1u32];

        let mut hybrid = FilteredScorer::new(
            QueryVector::NearestMasked(
                VectorInternal::Dense(query.clone()),
                dims,
                Some(1), // mask only layer 0
                false,
                None,
            ),
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            None,
            &deleted,
            HardwareCounterCell::new(),
        )
        .unwrap();

        let full = new_raw_scorer(
            QueryVector::Nearest(VectorInternal::Dense(query.clone())),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let masked = new_raw_scorer(
            QueryVector::NearestMasked(VectorInternal::Dense(query), vec![0, 1], None, false, None),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();

        // Layer 0 (< 1) → masked
        hybrid.set_layer(0);
        for i in 0..points.len() as PointOffsetType {
            assert!(
                (hybrid.score_point(i) - masked.score_point(i)).abs() < 1e-4,
                "layer 0 should use masked distance at point {i}",
            );
        }

        // Layer 1+ (≥ 1) → full
        hybrid.set_layer(1);
        for i in 0..points.len() as PointOffsetType {
            assert!(
                (hybrid.score_point(i) - full.score_point(i)).abs() < 1e-4,
                "layer 1 should use full distance at point {i}",
            );
        }
        hybrid.set_layer(5);
        for i in 0..points.len() as PointOffsetType {
            assert!((hybrid.score_point(i) - full.score_point(i)).abs() < 1e-4);
        }
    }

    #[test]
    fn alpha_zero_matches_masked_on_all_layers() {
        use common::bitvec::BitVec;

        use crate::index::hnsw_index::point_scorer::FilteredScorer;

        let points = vec![
            vec![1.0, 0.0, 100.0],
            vec![0.0, 1.0, 100.0],
            vec![1.0, 1.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        let dims = vec![0u32, 1u32];

        let mut blend = FilteredScorer::new(
            QueryVector::NearestMasked(
                VectorInternal::Dense(query.clone()),
                dims.clone(),
                None,
                false,
                Some(ordered_float::OrderedFloat(0.0)),
            ),
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            None,
            &deleted,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let masked = new_raw_scorer(
            QueryVector::NearestMasked(VectorInternal::Dense(query), dims, None, false, None),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();

        for layer in [0usize, 1, 5] {
            blend.set_layer(layer);
            for i in 0..points.len() as PointOffsetType {
                assert!(
                    (blend.score_point(i) - masked.score_point(i)).abs() < 1e-4,
                    "alpha=0 layer {layer} point {i}",
                );
            }
        }
    }

    #[test]
    fn alpha_one_matches_full_on_all_layers() {
        use common::bitvec::BitVec;

        use crate::index::hnsw_index::point_scorer::FilteredScorer;

        let points = vec![
            vec![1.0, 0.0, 100.0],
            vec![0.0, 1.0, 100.0],
            vec![1.0, 1.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        let dims = vec![0u32, 1u32];

        let mut blend = FilteredScorer::new(
            QueryVector::NearestMasked(
                VectorInternal::Dense(query.clone()),
                dims,
                None,
                false,
                Some(ordered_float::OrderedFloat(1.0)),
            ),
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            None,
            &deleted,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let full = new_raw_scorer(
            QueryVector::Nearest(VectorInternal::Dense(query)),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();

        for layer in [0usize, 3] {
            blend.set_layer(layer);
            for i in 0..points.len() as PointOffsetType {
                assert!(
                    (blend.score_point(i) - full.score_point(i)).abs() < 1e-4,
                    "alpha=1 layer {layer} point {i}",
                );
            }
        }
    }

    #[test]
    fn hybrid_alpha_blends_only_masked_layers() {
        use common::bitvec::BitVec;

        use crate::index::hnsw_index::point_scorer::FilteredScorer;

        let points = vec![
            vec![1.0, 0.0, 100.0],
            vec![0.0, 1.0, 100.0],
            vec![1.0, 1.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        let dims = vec![0u32, 1u32];
        let alpha = 0.25_f32;

        let mut hybrid = FilteredScorer::new(
            QueryVector::NearestMasked(
                VectorInternal::Dense(query.clone()),
                dims.clone(),
                Some(1),
                false,
                Some(ordered_float::OrderedFloat(alpha)),
            ),
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            None,
            &deleted,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let full = new_raw_scorer(
            QueryVector::Nearest(VectorInternal::Dense(query.clone())),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let masked = new_raw_scorer(
            QueryVector::NearestMasked(VectorInternal::Dense(query), dims, None, false, None),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();

        // Layer 0 (< 1) → blend
        hybrid.set_layer(0);
        for i in 0..points.len() as PointOffsetType {
            let expected = alpha * full.score_point(i) + (1.0 - alpha) * masked.score_point(i);
            assert!(
                (hybrid.score_point(i) - expected).abs() < 1e-4,
                "blended layer 0 point {i}",
            );
        }

        // Layer ≥ 1 → full only (no blend)
        hybrid.set_layer(1);
        for i in 0..points.len() as PointOffsetType {
            assert!(
                (hybrid.score_point(i) - full.score_point(i)).abs() < 1e-4,
                "full layer 1 point {i}",
            );
        }
    }

    #[test]
    fn masked_rejects_sparse_query_vector() {
        use sparse::common::sparse_vector::SparseVector;

        let storage = build_storage(Distance::Dot, &[vec![1.0, 2.0, 3.0]]);
        let err = new_raw_scorer(
            QueryVector::NearestMasked(
                VectorInternal::Sparse(SparseVector::new(vec![0], vec![1.0]).unwrap()),
                vec![0],
                None,
                false,
                None,
            ),
            &storage,
            HardwareCounterCell::new(),
        );
        assert!(err.is_err());
    }

    #[test]
    fn gather_kernel_matches_repack_on_scorer() {
        use crate::spaces::simple::DotProductMetric;
        use crate::vector_storage::query_scorer::QueryScorer;
        use crate::vector_storage::query_scorer::masked_kernels::MaskedKernelMode;
        use crate::vector_storage::query_scorer::masked_metric_query_scorer::MaskedMetricQueryScorer;

        let points = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let query = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let dims = [0u32, 2, 4, 7];

        let dense = match &storage {
            crate::vector_storage::VectorStorageEnum::DenseVolatile(vs) => vs,
            _ => panic!("expected volatile dense"),
        };

        let repack = MaskedMetricQueryScorer::<_, DotProductMetric, _>::new_with_kernel(
            query.clone(),
            &dims,
            dense,
            HardwareCounterCell::new(),
            MaskedKernelMode::Repack,
        );
        let gather = MaskedMetricQueryScorer::<_, DotProductMetric, _>::new_with_kernel(
            query,
            &dims,
            dense,
            HardwareCounterCell::new(),
            MaskedKernelMode::Gather,
        );

        for i in 0..points.len() as PointOffsetType {
            let a = QueryScorer::score_stored(&repack, i);
            let b = QueryScorer::score_stored(&gather, i);
            assert!((a - b).abs() < 1e-4, "point {i}: repack={a} gather={b}");
        }
    }
}
