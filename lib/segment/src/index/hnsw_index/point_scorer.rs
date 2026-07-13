use std::sync::atomic::AtomicBool;

use common::bitvec::BitSlice;
use common::condition_checker::ConditionChecker;
use common::counter::hardware_counter::HardwareCounterCell;
use common::cow::SimpleCow;
use common::fixed_length_priority_queue::FixedLengthPriorityQueue;
use common::generic_consts::Random;
use common::types::{PointOffsetType, ScoreType, ScoredPointOffset};
use smallvec::SmallVec;

use crate::common::operation_error::{OperationResult, check_process_stopped};
use crate::data_types::vectors::QueryVector;
use crate::index::query_optimization::optimized_filter::OptimizedFilter;
use crate::vector_storage::common::VECTOR_READ_BATCH_SIZE;
use crate::vector_storage::quantized::quantized_query_scorer::InternalScorerUnsupported;
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectorsRead;
use crate::vector_storage::query_scorer::QueryScorerBytes;
use crate::vector_storage::{
    RawScorer, RawScorerBuilder, VectorStorageRead, check_deleted_condition,
};
#[cfg(feature = "testing")]
use crate::vector_storage::{VectorStorageEnum, new_raw_scorer};

/// Scorers composition:
///
/// ```plaintext
///                                                               Metric
///                                                              ┌─────────────┐
///                                                              │ - Cosine    │
///  FilteredScorer      RawScorer          QueryScorer          │ - Dot       │
/// ┌─────────────────┐ ┌───────────────┐   ┌────────────────┐ ┌─┤ - Euclidean │
/// │ RawScorer ◄─────┼─┤ QueryScorer ◄─┼───│ Metric ◄───────┼─┘ └─────────────┘
/// │                 │ └───────────────┘   │                │    - Vector Distance
/// │ ConditionChecker│  - Access patterns  │ Query  ◄───────┼─┐
/// │                 │                     │                │ │  Query
/// │ deleted_points  │                     │ TVectorStorage │ │ ┌──────────────────┐
/// │ deleted_vectors │                     └────────────────┘ └─┤ - RecoQuery      │
/// └─────────────────┘                                          │ - DiscoverQuery  │
///                                                              │ - ContextQuery   │
///                                                              └──────────────────┘
///                                                              - Scoring logic
///                                                              - Complex queries
/// ```
///
/// The `BatchFilteredSearcher` contains an array of `RawScorer`s, a common filter and certain parameters.
///
/// ```plaintext
/// BatchFilteredSearcher  RawScorer
///  ┌─────────────────┐  ┌───────────────┐
///  │ [RawScorer] ◄───┼──┤ QueryScorer ◄─┼── (ditto)
///  │                 │  └───────────────┘
///  │ ConditionChecker│
///  └─────────────────┘
/// ```
pub struct FilteredScorer<'a> {
    /// Primary scorer. For hybrid masked search this is the *full*-distance scorer;
    /// otherwise it is the sole scorer used for the whole traversal.
    raw_scorer: Box<dyn RawScorer + 'a>,
    /// Optional masked-distance scorer used on layers `current_layer < mask_from_layer`.
    masked_layer_scorer: Option<Box<dyn RawScorer + 'a>>,
    /// Layers strictly below this cutoff use [`Self::masked_layer_scorer`].
    /// Ignored when `masked_layer_scorer` is `None`.
    mask_from_layer: usize,
    /// Current HNSW layer being searched. Updated via [`Self::set_layer`].
    current_layer: usize,
    filters: ScorerFilters<'a>,
    /// Temporary buffer for scores.
    scores_buffer: Vec<ScoreType>,
}

pub struct ScorerFilters<'a> {
    filter_context: Option<FilterContext<'a>>,
    /// Point deleted flags should be explicitly present as `false`
    /// for each existing point in the segment.
    /// If there are no flags for some points, they are considered deleted.
    /// [`BitSlice`] defining flags for deleted points (and thus these vectors).
    point_deleted: &'a BitSlice,
    /// [`BitSlice`] defining flags for deleted vectors in this segment.
    vec_deleted: &'a BitSlice,
}

type FilterContext<'a> = SimpleCow<'a, OptimizedFilter<'a>>;

impl<'a> ScorerFilters<'a> {
    /// Return true if vector satisfies current search context for given point:
    /// exists, not deleted, and satisfies filter context.
    pub fn check_vector(&self, point_id: PointOffsetType) -> bool {
        check_deleted_condition(point_id, self.vec_deleted, self.point_deleted)
            && self
                .filter_context
                .as_ref()
                .is_none_or(|f| f.check_infallible(point_id))
    }

    fn as_borrowed(&'a self) -> Self {
        ScorerFilters {
            filter_context: self.filter_context.as_ref().map(SimpleCow::as_borrowed),
            point_deleted: self.point_deleted,
            vec_deleted: self.vec_deleted,
        }
    }
}

pub struct FilteredBytesScorer<'a> {
    scorer_bytes: &'a dyn QueryScorerBytes,
    filters: ScorerFilters<'a>,
}

impl<'a> FilteredBytesScorer<'a> {
    pub fn score_points(
        &self,
        points: &mut Vec<(PointOffsetType, &[u8])>,
        limit: usize,
    ) -> impl Iterator<Item = ScoredPointOffset> {
        points.retain(|(point_id, _)| self.filters.check_vector(*point_id));
        if limit != 0 {
            points.truncate(limit);
        }

        points.iter().map(|&(idx, bytes)| ScoredPointOffset {
            idx,
            score: self.scorer_bytes.score_bytes(bytes),
        })
    }
}

impl<'a> FilteredScorer<'a> {
    /// Create a new filtered scorer.
    ///
    /// If present, `quantized_vectors` will be used for scoring, otherwise `vectors` will be used.
    ///
    /// For hybrid masked queries (`NearestMasked` with `Some(mask_from_layer)` where the cutoff
    /// is > 0), both a full-distance and a masked-distance scorer are built. Call
    /// [`Self::set_layer`] before scoring each HNSW layer so the correct scorer is selected.
    pub fn new<V, Q>(
        query: QueryVector,
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        filter_context: Option<FilterContext<'a>>,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        let filters = ScorerFilters {
            filter_context,
            point_deleted,
            vec_deleted: vectors.deleted_vector_bitslice(),
        };

        // Hybrid Option 2: full distance on coarse layers, masked on bottom layers.
        // Build both scorers once; layer selection happens via `set_layer` (no per-compare branch
        // beyond a single integer compare + pointer pick).
        if let QueryVector::NearestMasked(vector, dims, Some(mask_from_layer)) = &query
            && *mask_from_layer > 0
        {
            if quantized_vectors.is_some() {
                return Err(
                    crate::common::operation_error::OperationError::validation_error(
                        "masked focus search is not supported with quantization",
                    ),
                );
            }
            let full_scorer = vectors.build_raw_scorer(
                QueryVector::Nearest(vector.clone()),
                hardware_counter.fork(),
            )?;
            let masked_scorer = vectors.build_raw_scorer(
                QueryVector::NearestMasked(vector.clone(), dims.clone(), None),
                hardware_counter,
            )?;
            return Ok(FilteredScorer {
                raw_scorer: full_scorer,
                masked_layer_scorer: Some(masked_scorer),
                mask_from_layer: *mask_from_layer,
                current_layer: 0,
                filters,
                scores_buffer: Vec::new(),
            });
        }

        let raw_scorer = match quantized_vectors {
            Some(quantized_vectors) => quantized_vectors.raw_scorer(query, hardware_counter)?,
            None => vectors.build_raw_scorer(query, hardware_counter)?,
        };
        Ok(FilteredScorer {
            raw_scorer,
            masked_layer_scorer: None,
            mask_from_layer: 0,
            current_layer: 0,
            filters,
            scores_buffer: Vec::new(),
        })
    }

    pub fn new_internal<V, Q>(
        point_id: PointOffsetType,
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        filter_context: Option<FilterContext<'a>>,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        // This is a fallback function, which is used if quantized vector storage
        // is not capable of reconstructing the query vector.
        let original_query_fn = || {
            let query = vectors.get_vector::<Random>(point_id);
            let query: QueryVector = query.as_vec_ref().into();
            query
        };
        let raw_scorer = match quantized_vectors {
            Some(quantized_vectors) => quantized_vectors
                .raw_internal_scorer(point_id, hardware_counter)
                .or_else(|InternalScorerUnsupported(hardware_counter)| {
                    quantized_vectors.raw_scorer(original_query_fn(), hardware_counter)
                })?,
            None => {
                let query = original_query_fn();
                vectors.build_raw_scorer(query, hardware_counter)?
            }
        };
        Ok(FilteredScorer {
            raw_scorer,
            masked_layer_scorer: None,
            mask_from_layer: 0,
            current_layer: 0,
            filters: ScorerFilters {
                filter_context,
                point_deleted,
                vec_deleted: vectors.deleted_vector_bitslice(),
            },
            scores_buffer: Vec::new(),
        })
    }

    /// Create a new filtered scorer for testing purposes.
    ///
    /// # Panics
    ///
    /// Panics if [`new_raw_scorer`] fails. Does not set up hybrid layer scorers;
    /// use [`Self::new`] for Option-2 hybrid masked queries in tests.
    #[cfg(feature = "testing")]
    pub fn new_for_test(
        vector: QueryVector,
        vector_storage: &'a VectorStorageEnum,
        point_deleted: &'a BitSlice,
    ) -> Self {
        FilteredScorer {
            raw_scorer: new_raw_scorer(vector, vector_storage, HardwareCounterCell::new()).unwrap(),
            masked_layer_scorer: None,
            mask_from_layer: 0,
            current_layer: 0,
            filters: ScorerFilters {
                filter_context: None,
                point_deleted,
                vec_deleted: vector_storage.deleted_vector_bitslice(),
            },
            scores_buffer: Vec::new(),
        }
    }

    /// Select the active distance function for the given HNSW layer.
    ///
    /// For hybrid masked search, layers `L < mask_from_layer` use masked distance and
    /// coarser layers use full distance. No-op when no hybrid scorer is configured.
    #[inline]
    pub fn set_layer(&mut self, layer: usize) {
        self.current_layer = layer;
    }

    #[inline]
    fn active_raw_scorer(&self) -> &dyn RawScorer {
        if let Some(masked) = self.masked_layer_scorer.as_ref()
            && self.current_layer < self.mask_from_layer
        {
            return masked.as_ref();
        }
        self.raw_scorer.as_ref()
    }

    pub fn raw_scorer(&self) -> &dyn RawScorer {
        self.active_raw_scorer()
    }

    pub fn filters(&self) -> &ScorerFilters<'a> {
        &self.filters
    }

    /// Return [`FilteredBytesScorer`] if the underlying scorer supports it.
    pub fn scorer_bytes(&self) -> Option<FilteredBytesScorer<'_>> {
        Some(FilteredBytesScorer {
            scorer_bytes: self.active_raw_scorer().scorer_bytes()?,
            filters: self.filters.as_borrowed(),
        })
    }

    /// Filters and calculates scores for the given slice of points IDs.
    ///
    /// For performance reasons this method mutates `point_ids`.
    ///
    /// # Arguments
    ///
    /// * `point_ids` - list of points to score.
    ///   **Warning**: This input will be wrecked during the execution.
    /// * `limit` - limits the number of points to process after filtering.
    ///   `0` means no limit.
    pub fn score_points(
        &mut self,
        point_ids: &mut Vec<PointOffsetType>,
        limit: usize,
    ) -> impl Iterator<Item = ScoredPointOffset> {
        point_ids.retain(|point_id| self.filters.check_vector(*point_id));
        if limit != 0 {
            point_ids.truncate(limit);
        }

        self.score_points_unfiltered(point_ids)
    }

    pub fn score_points_unfiltered(
        &mut self,
        point_ids: &[PointOffsetType],
    ) -> impl Iterator<Item = ScoredPointOffset> {
        if self.scores_buffer.len() < point_ids.len() {
            self.scores_buffer.resize(point_ids.len(), 0.0);
        }

        let scores = &mut self.scores_buffer[..point_ids.len()];
        let use_masked =
            self.masked_layer_scorer.is_some() && self.current_layer < self.mask_from_layer;
        if use_masked {
            self.masked_layer_scorer
                .as_ref()
                .unwrap()
                .score_points(point_ids, scores);
        } else {
            self.raw_scorer.score_points(point_ids, scores);
        }

        std::iter::zip(point_ids, &self.scores_buffer)
            .map(|(&idx, &score)| ScoredPointOffset { idx, score })
    }

    pub fn score_point(&self, point_id: PointOffsetType) -> ScoreType {
        self.active_raw_scorer().score_point(point_id)
    }

    pub fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        self.active_raw_scorer().score_internal(point_a, point_b)
    }
}

// We keep each scorer with its queue to reduce allocations and improve data locality.
struct BatchSearch<'a> {
    raw_scorer: Box<dyn RawScorer + 'a>,
    pq: FixedLengthPriorityQueue<ScoredPointOffset>,
}

pub struct BatchFilteredSearcher<'a> {
    scorer_batch: SmallVec<[BatchSearch<'a>; 1]>,
    filters: ScorerFilters<'a>,
}

impl<'a> BatchFilteredSearcher<'a> {
    /// Create a new batch filtered searcher.
    ///
    /// If present, `quantized_vectors` will be used for scoring, otherwise `vectors` will be used.
    pub fn new<V, Q>(
        queries: &[&QueryVector],
        vectors: &'a V,
        quantized_vectors: Option<&'a Q>,
        filter_context: Option<FilterContext<'a>>,
        top: usize,
        point_deleted: &'a BitSlice,
        hardware_counter: HardwareCounterCell,
    ) -> OperationResult<Self>
    where
        V: VectorStorageRead + RawScorerBuilder,
        Q: QuantizedVectorsRead,
    {
        let scorer_batch = queries
            .iter()
            .map(|&query| {
                let query = query.to_owned();
                let hardware_counter = hardware_counter.fork();
                let raw_scorer = match quantized_vectors {
                    Some(quantized_vectors) => {
                        quantized_vectors.raw_scorer(query, hardware_counter)
                    }
                    None => vectors.build_raw_scorer(query, hardware_counter),
                };
                let pq = FixedLengthPriorityQueue::new(top);
                raw_scorer.map(|raw_scorer| BatchSearch { raw_scorer, pq })
            })
            .collect::<Result<_, _>>()?;
        let filters = ScorerFilters {
            filter_context,
            point_deleted,
            vec_deleted: vectors.deleted_vector_bitslice(),
        };
        Ok(Self {
            scorer_batch,
            filters,
        })
    }

    /// Create a new batched filtered searcher for testing purposes.
    ///
    /// # Panics
    ///
    /// Panics if [`new_raw_scorer`] fails.
    #[cfg(feature = "testing")]
    pub fn new_for_test(
        vectors: &[QueryVector],
        vector_storage: &'a VectorStorageEnum,
        point_deleted: &'a BitSlice,
        top: usize,
    ) -> Self {
        let scorer_batch = vectors
            .iter()
            .map(|vector| {
                let raw_scorer = new_raw_scorer(
                    vector.to_owned(),
                    vector_storage,
                    HardwareCounterCell::new(),
                )
                .unwrap();
                BatchSearch {
                    raw_scorer,
                    pq: FixedLengthPriorityQueue::new(top),
                }
            })
            .collect();
        Self {
            scorer_batch,
            filters: ScorerFilters {
                filter_context: None,
                point_deleted,
                vec_deleted: vector_storage.deleted_vector_bitslice(),
            },
        }
    }

    /// Iterator over every internal point ID that isn't soft-deleted in this
    /// searcher's `point_deleted` bitslice.
    ///
    /// Does not apply deferred-point filtering — wrap with
    /// `PointMappingsRefEnum::filter_deferred_and_deleted` (or compose otherwise) before
    /// passing to [`Self::peek_top_iter`] when deferred awareness is needed.
    ///
    /// The returned iterator borrows the underlying bitslice (lifetime `'a`),
    /// independent of `&self`, so it can be composed and then passed into
    /// `peek_top_iter(self, ...)` which consumes the searcher.
    pub fn iter_not_deleted(&self) -> impl Iterator<Item = PointOffsetType> + 'a {
        self.filters
            .point_deleted
            .iter_zeros()
            .map(|p| p as PointOffsetType)
    }

    /// Score every non-deleted point without deferred filtering.
    ///
    /// Production paths compose `iter_not_deleted` with
    /// `PointMappingsRefEnum::filter_deferred_and_deleted` and call
    /// [`Self::peek_top_iter`] directly.
    #[cfg(feature = "testing")]
    pub fn peek_top_all(
        self,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        let iter = self.iter_not_deleted();
        self.peek_top_iter(iter, is_stopped)
    }

    /// This function expects deferred points to be already filtered from the iterator.
    pub fn peek_top_iter(
        mut self,
        mut points: impl Iterator<Item = PointOffsetType>,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<Vec<ScoredPointOffset>>> {
        // Reuse the same buffer for all chunks, to avoid reallocation
        let mut chunk = [0; VECTOR_READ_BATCH_SIZE];
        let mut scores_buffer = [0.0; VECTOR_READ_BATCH_SIZE];

        loop {
            check_process_stopped(is_stopped)?;

            let mut chunk_size = 0;
            for point_id in &mut points {
                check_process_stopped(is_stopped)?;

                if !self.filters.check_vector(point_id) {
                    continue;
                }
                chunk[chunk_size] = point_id;
                chunk_size += 1;
                if chunk_size == VECTOR_READ_BATCH_SIZE {
                    break;
                }
            }

            if chunk_size == 0 {
                break;
            }

            // Switching the loops improves batching performance, but slightly degrades single-query performance.
            for BatchSearch { raw_scorer, pq } in &mut self.scorer_batch {
                raw_scorer.score_points(&chunk[..chunk_size], &mut scores_buffer[..chunk_size]);

                for i in 0..chunk_size {
                    pq.push(ScoredPointOffset {
                        idx: chunk[i],
                        score: scores_buffer[i],
                    });
                }
            }
        }

        let results = self
            .scorer_batch
            .into_iter()
            .map(|BatchSearch { pq, .. }| pq.into_sorted_vec())
            .collect();
        Ok(results)
    }
}
