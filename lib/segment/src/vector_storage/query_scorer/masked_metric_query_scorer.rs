use std::borrow::Cow;
use std::cell::RefCell;
use std::marker::PhantomData;

use common::counter::hardware_counter::HardwareCounterCell;
use common::generic_consts::Random;
use common::typelevel::False;
use common::types::{PointOffsetType, ScoreType};

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
    /// Avoids per-comparison allocations while reusing the (SIMD-accelerated) metric.
    buffer: RefCell<TypedDenseVector<TElement>>,
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
        mut hardware_counter: HardwareCounterCell,
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
            metric: PhantomData,
            hardware_counter,
        }
    }

    /// Gather the focus dimensions of `vector` and score them against the masked query.
    #[inline]
    fn score_masked(&self, vector: &[TElement]) -> ScoreType {
        let mut buffer = self.buffer.borrow_mut();
        for (slot, &dim) in buffer.iter_mut().zip(self.dims.iter()) {
            *slot = vector[dim];
        }
        TMetric::similarity(&self.query, &buffer)
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

        let masked_a: TypedDenseVector<TElement> = self.dims.iter().map(|&dim| v1[dim]).collect();
        let mut buffer = self.buffer.borrow_mut();
        for (slot, &dim) in buffer.iter_mut().zip(self.dims.iter()) {
            *slot = v2[dim];
        }
        TMetric::similarity(&masked_a, &buffer)
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
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::dense::volatile_dense_vector_storage::new_volatile_dense_vector_storage;
    use crate::vector_storage::new_raw_scorer;

    fn build_storage(distance: Distance, points: &[Vec<f32>]) -> crate::vector_storage::VectorStorageEnum {
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
        let masked = QueryVector::NearestMasked(VectorInternal::Dense(query.clone()), dims.clone());
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
                QueryVector::NearestMasked(VectorInternal::Dense(query.clone()), all_dims),
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
        );
        let scorer = new_raw_scorer(masked, &storage, HardwareCounterCell::new()).unwrap();

        assert!((scorer.score_point(0) - 2.0).abs() < 1e-4);
        assert!((scorer.score_point(1) - 2.0).abs() < 1e-4);
    }
}
