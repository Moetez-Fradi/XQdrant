use common::bitvec::BitSlice;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::ScoredPointOffset;
use itertools::Itertools;

use crate::common::operation_error::OperationResult;
use crate::data_types::vectors::QueryVector;
use crate::index::hnsw_index::point_scorer::FilteredScorer;
use crate::types::{
    SearchParams, default_quantization_ignore_value, default_quantization_oversampling_value,
};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectorsRead;
use crate::vector_storage::{RawScorerBuilder, VectorStorageRead};

pub fn is_quantized_search<Q: QuantizedVectorsRead>(
    quantized_storage: Option<&Q>,
    params: Option<&SearchParams>,
) -> bool {
    let ignore_quantization = params
        .and_then(|p| p.quantization)
        .map(|q| q.ignore)
        .unwrap_or(default_quantization_ignore_value());
    let exact = params.is_some_and(|p| p.exact);
    quantized_storage.is_some() && !ignore_quantization && !exact
}

pub fn get_oversampled_top<Q: QuantizedVectorsRead>(
    quantized_storage: Option<&Q>,
    params: Option<&SearchParams>,
    top: usize,
) -> usize {
    let quantization_enabled = is_quantized_search(quantized_storage, params);

    let oversampling_value = params
        .and_then(|p| p.quantization)
        .map(|q| q.oversampling)
        .unwrap_or(default_quantization_oversampling_value());

    match oversampling_value {
        Some(oversampling) if quantization_enabled && oversampling > 1.0 => {
            (oversampling * top as f64) as usize
        }
        _ => top,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn postprocess_search_result<V, Q>(
    mut search_result: Vec<ScoredPointOffset>,
    point_deleted: &BitSlice,
    vector_storage: &V,
    quantized_vectors: Option<&Q>,
    vector: &QueryVector,
    params: Option<&SearchParams>,
    top: usize,
    hardware_counter: HardwareCounterCell,
) -> OperationResult<Vec<ScoredPointOffset>>
where
    V: VectorStorageRead + RawScorerBuilder,
    Q: QuantizedVectorsRead,
{
    let quantization_enabled = is_quantized_search(quantized_vectors, params);

    let default_rescoring = quantized_vectors
        .as_ref()
        .map(|q| q.default_rescoring())
        .unwrap_or(false);
    let quantization_rescore = quantization_enabled
        && params
            .and_then(|p| p.quantization)
            .and_then(|q| q.rescore)
            .unwrap_or(default_rescoring);

    // X3 verify: after masked / hybrid traversal, re-score candidates with full-vector distance.
    // Mutually exclusive with quantization rescoring (masked already rejects quantization).
    let verify_rescore = matches!(vector, QueryVector::NearestMasked(_, _, _, true, _));

    if quantization_rescore || verify_rescore {
        let rescore_query = if let QueryVector::NearestMasked(inner, _, _, true, _) = vector {
            QueryVector::Nearest(inner.clone())
        } else {
            // Quantization rescore: score with the original (typically Nearest) query on full vectors.
            vector.clone()
        };
        let mut scorer = FilteredScorer::new(
            rescore_query,
            vector_storage,
            // Always use full (non-quantized) vectors for verify / quantization rescore.
            None::<&Q>,
            None,
            point_deleted,
            hardware_counter,
        )?;

        search_result = scorer
            .score_points(&mut search_result.iter().map(|x| x.idx).collect_vec(), 0)
            .collect();
        search_result.sort_unstable();
        search_result.reverse();
    }
    search_result.truncate(top);
    Ok(search_result)
}

#[cfg(test)]
mod tests {
    use common::bitvec::BitVec;
    use common::counter::hardware_counter::HardwareCounterCell;
    use common::types::{PointOffsetType, ScoredPointOffset};

    use super::postprocess_search_result;
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
    fn verify_false_leaves_masked_scores_unchanged() {
        let points = vec![
            vec![1.0, 0.0, 100.0],
            vec![0.0, 1.0, 100.0],
            vec![1.0, 1.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        let dims = vec![0u32, 1u32];

        let masked_query = QueryVector::NearestMasked(
            VectorInternal::Dense(query.clone()),
            dims,
            None,
            false,
            None,
        );
        let masked_scorer =
            new_raw_scorer(masked_query.clone(), &storage, HardwareCounterCell::new()).unwrap();

        let candidates: Vec<ScoredPointOffset> = (0..points.len() as PointOffsetType)
            .map(|idx| ScoredPointOffset {
                idx,
                score: masked_scorer.score_point(idx),
            })
            .collect();

        let out = postprocess_search_result(
            candidates.clone(),
            &deleted,
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            &masked_query,
            None,
            candidates.len(),
            HardwareCounterCell::new(),
        )
        .unwrap();

        assert_eq!(out.len(), candidates.len());
        for (a, b) in out.iter().zip(candidates.iter()) {
            assert_eq!(a.idx, b.idx);
            assert!((a.score - b.score).abs() < 1e-5);
        }
    }

    #[test]
    fn verify_true_returns_full_distance_scores_and_can_reorder() {
        // Point 0: high masked score on dims [0,1], low full score (large negative dim 2).
        // Point 1: lower masked score, but much higher full score.
        let points = vec![
            vec![10.0, 10.0, -1000.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        let dims = vec![0u32, 1u32];

        let masked_scorer = new_raw_scorer(
            QueryVector::NearestMasked(
                VectorInternal::Dense(query.clone()),
                dims.clone(),
                None,
                false,
                None,
            ),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let full_scorer = new_raw_scorer(
            QueryVector::Nearest(VectorInternal::Dense(query.clone())),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();

        let masked_ranked: Vec<ScoredPointOffset> = {
            let mut pts: Vec<_> = (0..points.len() as PointOffsetType)
                .map(|idx| ScoredPointOffset {
                    idx,
                    score: masked_scorer.score_point(idx),
                })
                .collect();
            pts.sort_unstable();
            pts.reverse();
            pts
        };

        // Masked ranking should prefer point 0 over point 1.
        assert_eq!(masked_ranked[0].idx, 0);

        let verify_query =
            QueryVector::NearestMasked(VectorInternal::Dense(query), dims, None, true, None);
        let verified = postprocess_search_result(
            masked_ranked.clone(),
            &deleted,
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            &verify_query,
            None,
            2,
            HardwareCounterCell::new(),
        )
        .unwrap();

        assert_eq!(verified.len(), 2);
        // Full-distance ranking prefers point 1.
        assert_eq!(verified[0].idx, 1);
        assert!((verified[0].score - full_scorer.score_point(1)).abs() < 1e-5);
        assert!((verified[1].score - full_scorer.score_point(verified[1].idx)).abs() < 1e-5);

        // Scores must be full-distance, not masked.
        assert!((verified[0].score - masked_scorer.score_point(1)).abs() > 1e-3);
    }

    #[test]
    fn verify_with_alpha_still_uses_full_distance_rescore() {
        // Alpha only affects traversal scoring; verify postprocess must ignore it.
        let points = vec![
            vec![100.0, 100.0, -1000.0],
            vec![10.0, 10.0, -1000.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];
        let storage = build_storage(Distance::Dot, &points);
        let deleted = BitVec::repeat(false, points.len());
        let query = vec![1.0, 1.0, 1.0];
        let dims = vec![0u32, 1u32];

        let masked_scorer = new_raw_scorer(
            QueryVector::NearestMasked(
                VectorInternal::Dense(query.clone()),
                dims.clone(),
                None,
                false,
                None,
            ),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();
        let full_scorer = new_raw_scorer(
            QueryVector::Nearest(VectorInternal::Dense(query.clone())),
            &storage,
            HardwareCounterCell::new(),
        )
        .unwrap();

        let candidates: Vec<ScoredPointOffset> = (0..points.len() as PointOffsetType)
            .map(|idx| ScoredPointOffset {
                idx,
                // Pretend traversal returned blended / masked-ish scores.
                score: 0.5 * full_scorer.score_point(idx) + 0.5 * masked_scorer.score_point(idx),
            })
            .collect();

        let verify_query = QueryVector::NearestMasked(
            VectorInternal::Dense(query),
            dims,
            None,
            true,
            Some(ordered_float::OrderedFloat(0.5)),
        );
        let verified = postprocess_search_result(
            candidates,
            &deleted,
            &storage,
            None::<&crate::vector_storage::quantized::quantized_vectors::QuantizedVectors>,
            &verify_query,
            None,
            2,
            HardwareCounterCell::new(),
        )
        .unwrap();

        assert_eq!(verified[0].idx, 2);
        assert!((verified[0].score - full_scorer.score_point(2)).abs() < 1e-5);
    }
}
