use ahash::AHashSet;
use api::rest::LookupLocation;
use common::types::ScoreType;
use itertools::Itertools;
use ordered_float::OrderedFloat;
use segment::data_types::order_by::OrderBy;
use segment::data_types::vectors::{DEFAULT_VECTOR_NAME, NamedQuery, VectorInternal, VectorRef};
use segment::index::query_optimization::rescore_formula::parsed_formula::ParsedFormula;
use segment::json_path::JsonPath;
use segment::types::{
    Condition, ExtendedPointId, Filter, HasIdCondition, PointIdType, SearchParams, VectorName,
    VectorNameBuf, WithPayloadInterface, WithVector,
};
use segment::vector_storage::query::{
    ContextPair, ContextQuery, DiscoverQuery, FeedbackItem, NaiveFeedbackCoefficients, RecoQuery,
};
use serde::Serialize;
use shard::query::query_enum::QueryEnum;

use super::formula::FormulaInternal;
use super::shard_query::{
    DimsExplainedInternal, DimsFocusInternal, FusionInternal, SampleInternal, ScoringQuery,
    ShardPrefetch, ShardQueryRequest,
};
use crate::common::fetch_vectors::ReferencedVectors;
use crate::lookup::WithLookup;
use crate::operations::types::{CollectionError, CollectionResult};
use crate::operations::universal_query::shard_query::MmrInternal;
use crate::recommendations::avg_vector_for_recommendation;

const DEFAULT_MMR_LAMBDA: f32 = 0.5;

/// Internal representation of a query request, used to converge from REST and gRPC. This can have IDs referencing vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectionQueryRequest {
    pub prefetch: Vec<CollectionPrefetch>,
    pub query: Option<Query>,
    pub using: VectorNameBuf,
    pub filter: Option<Filter>,
    pub score_threshold: Option<ScoreType>,
    pub limit: usize,
    pub offset: usize,
    /// Search params for when there is no prefetch
    pub params: Option<SearchParams>,
    pub with_vector: WithVector,
    pub with_payload: WithPayloadInterface,
    pub lookup_from: Option<LookupLocation>,
    /// Include per-dimension score explanations into the response.
    pub dims_explained: Option<DimsExplainedInternal>,
}

impl CollectionQueryRequest {
    pub const DEFAULT_LIMIT: usize = 10;

    pub const DEFAULT_GROUP_SIZE: usize = 3;

    pub const DEFAULT_OFFSET: usize = 0;

    pub const DEFAULT_WITH_VECTOR: WithVector = WithVector::Bool(false);

    pub const DEFAULT_WITH_PAYLOAD: WithPayloadInterface = WithPayloadInterface::Bool(false);
}

/// Lightweight representation of a query request to implement the [`RetrieveRequest`] trait.
///
/// [`RetrieveRequest`]: crate::common::retrieve_request_trait::RetrieveRequest
#[derive(Debug)]
pub struct CollectionQueryResolveRequest {
    pub referenced_ids: Vec<PointIdType>,
    pub lookup_from: Option<LookupLocation>,
    pub using: VectorNameBuf,
}

/// Internal representation of a group query request, used to converge from REST and gRPC.
#[derive(Debug)]
pub struct CollectionQueryGroupsRequest {
    pub prefetch: Vec<CollectionPrefetch>,
    pub query: Option<Query>,
    pub using: VectorNameBuf,
    pub filter: Option<Filter>,
    pub params: Option<SearchParams>,
    pub score_threshold: Option<ScoreType>,
    pub with_vector: WithVector,
    pub with_payload: WithPayloadInterface,
    pub lookup_from: Option<LookupLocation>,
    pub group_by: JsonPath,
    pub group_size: usize,
    pub limit: usize,
    pub with_lookup: Option<WithLookup>,
    pub dims_explained: Option<DimsExplainedInternal>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Query {
    /// Score points against some vector(s)
    Vector(VectorQuery<VectorInputInternal>),

    /// Reciprocal rank fusion
    Fusion(FusionInternal),

    /// Order by a payload field
    OrderBy(OrderBy),

    /// Score boosting via an arbitrary formula
    Formula(FormulaInternal),

    /// Sample points
    Sample(SampleInternal),
}

impl Query {
    pub fn try_into_scoring_query(
        self,
        ids_to_vectors: &ReferencedVectors,
        lookup_vector_name: &VectorName,
        lookup_collection: Option<&String>,
        using: VectorNameBuf,
        request_limit: usize,
    ) -> CollectionResult<ScoringQuery> {
        let scoring_query = match self {
            Query::Vector(vector_query) => {
                vector_query
                    // Homogenize the input into raw vectors
                    .ids_into_vectors(ids_to_vectors, lookup_vector_name, lookup_collection)?
                    .preprocess_vectors()
                    // Turn into QueryEnum
                    .into_scoring_query(using, request_limit)?
            }
            Query::Fusion(fusion) => ScoringQuery::Fusion(fusion),
            Query::OrderBy(order_by) => ScoringQuery::OrderBy(order_by),
            Query::Formula(formula) => ScoringQuery::Formula(ParsedFormula::try_from(formula)?),
            Query::Sample(sample) => ScoringQuery::Sample(sample),
        };

        Ok(scoring_query)
    }

    pub fn get_referenced_ids(&self) -> Vec<PointIdType> {
        match self {
            Self::Vector(vector_query) => vector_query
                .get_referenced_ids()
                .into_iter()
                .copied()
                .collect(),
            Self::Fusion(_) | Self::OrderBy(_) | Self::Formula(_) | Self::Sample(_) => Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VectorInputInternal {
    Id(PointIdType),
    Vector(VectorInternal),
}

impl VectorInputInternal {
    pub fn as_id(&self) -> Option<&PointIdType> {
        match self {
            VectorInputInternal::Id(id) => Some(id),
            VectorInputInternal::Vector(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VectorQuery<T> {
    Nearest(T),
    NearestWithMmr(NearestWithMmr<T>),
    NearestWithFocus(NearestWithFocus<T>),
    RecommendAverageVector(RecoQuery<T>),
    RecommendBestScore(RecoQuery<T>),
    RecommendSumScores(RecoQuery<T>),
    Discover(DiscoverQuery<T>),
    Context(ContextQuery<T>),
    Feedback(FeedbackInternal<T>),
}

impl<T> VectorQuery<T> {
    /// Iterate through all items, without any kind of structure
    pub fn flat_iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        match self {
            VectorQuery::Nearest(input) => Box::new(std::iter::once(input)),
            VectorQuery::NearestWithMmr(query) => Box::new(std::iter::once(&query.nearest)),
            VectorQuery::NearestWithFocus(query) => Box::new(std::iter::once(&query.nearest)),
            VectorQuery::RecommendAverageVector(query)
            | VectorQuery::RecommendBestScore(query)
            | VectorQuery::RecommendSumScores(query) => Box::new(query.flat_iter()),
            VectorQuery::Discover(query) => Box::new(query.flat_iter()),
            VectorQuery::Context(query) => Box::new(query.flat_iter()),
            VectorQuery::Feedback(query) => Box::new(query.flat_iter()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NearestWithMmr<T> {
    pub nearest: T,
    pub mmr: Mmr,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Mmr {
    pub diversity: Option<f32>,
    pub candidates_limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NearestWithFocus<T> {
    pub nearest: T,
    pub focus: DimsFocus,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DimsFocus {
    /// Dimension indexes to focus the scoring on.
    pub dims: Vec<u32>,
    /// Maximum number of candidates to preselect using nearest neighbors.
    pub candidates_limit: Option<usize>,
    /// When true, compute the masked distance (over `dims` only) directly inside the HNSW
    /// traversal instead of preselecting full-vector candidates and re-scoring them.
    ///
    /// This is a real per-comparison speedup, but the results are approximate: the index
    /// graph was built for full vectors, so `candidates_limit` no longer applies.
    pub masked: bool,
    /// Hybrid cutoff: layer `L` uses masked distance when `L < mask_from_layer`, else full.
    /// Only meaningful with `masked = true`. `None` keeps legacy all-layers-masked behaviour.
    pub mask_from_layer: Option<u32>,
    /// After masked / hybrid traversal, re-score the returned top-k with full-vector distance.
    /// Only meaningful with `masked = true`. Returned scores are then full-distance scores.
    pub verify: bool,
    /// Blend full and masked similarity scores on masked layers:
    /// `alpha * s_full + (1 - alpha) * s_focus`. Must be in `[0, 1]`.
    /// Only meaningful with `masked = true`. `None` = no blend.
    pub alpha: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackInternal<T> {
    pub target: T,
    pub feedback: Vec<FeedbackItem<T>>,
    pub strategy: FeedbackStrategy,
}

impl<T> FeedbackInternal<T> {
    fn flat_iter(&self) -> impl Iterator<Item = &T> {
        self.feedback
            .iter()
            .map(|item| &item.vector)
            .chain(std::iter::once(&self.target))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FeedbackStrategy {
    Naive { a: f32, b: f32, c: f32 },
}

impl VectorQuery<VectorInputInternal> {
    /// Turns all [VectorInputInternal]s into [VectorInternal]s, using the provided [ReferencedVectors] to look up the vectors.
    ///
    /// Will panic if the ids are not found in the [ReferencedVectors].
    fn ids_into_vectors(
        self,
        ids_to_vectors: &ReferencedVectors,
        lookup_vector_name: &VectorName,
        lookup_collection: Option<&String>,
    ) -> CollectionResult<VectorQuery<VectorInternal>> {
        match self {
            VectorQuery::Nearest(vector_input) => {
                let vector = ids_to_vectors
                    .resolve_reference(lookup_collection, lookup_vector_name, vector_input)
                    .ok_or_else(|| vector_not_found_error(lookup_vector_name))?;

                Ok(VectorQuery::Nearest(vector))
            }
            VectorQuery::RecommendAverageVector(reco) => {
                let (positives, negatives) = Self::resolve_reco_reference(
                    reco,
                    ids_to_vectors,
                    lookup_vector_name,
                    lookup_collection,
                );
                Ok(VectorQuery::RecommendAverageVector(RecoQuery::new(
                    positives, negatives,
                )))
            }
            VectorQuery::RecommendBestScore(reco) => {
                let (positives, negatives) = Self::resolve_reco_reference(
                    reco,
                    ids_to_vectors,
                    lookup_vector_name,
                    lookup_collection,
                );
                Ok(VectorQuery::RecommendBestScore(RecoQuery::new(
                    positives, negatives,
                )))
            }
            VectorQuery::RecommendSumScores(reco) => {
                let (positives, negatives) = Self::resolve_reco_reference(
                    reco,
                    ids_to_vectors,
                    lookup_vector_name,
                    lookup_collection,
                );
                Ok(VectorQuery::RecommendSumScores(RecoQuery::new(
                    positives, negatives,
                )))
            }
            VectorQuery::Discover(discover) => {
                let target = ids_to_vectors
                    .resolve_reference(lookup_collection, lookup_vector_name, discover.target)
                    .ok_or_else(|| vector_not_found_error(lookup_vector_name))?;
                let pairs = discover
                    .pairs
                    .into_iter()
                    .map(|pair| {
                        Ok(ContextPair {
                            positive: ids_to_vectors
                                .resolve_reference(
                                    lookup_collection,
                                    lookup_vector_name,
                                    pair.positive,
                                )
                                .ok_or_else(|| vector_not_found_error(lookup_vector_name))?,
                            negative: ids_to_vectors
                                .resolve_reference(
                                    lookup_collection,
                                    lookup_vector_name,
                                    pair.negative,
                                )
                                .ok_or_else(|| vector_not_found_error(lookup_vector_name))?,
                        })
                    })
                    .collect::<CollectionResult<_>>()?;

                Ok(VectorQuery::Discover(DiscoverQuery { target, pairs }))
            }
            VectorQuery::Context(context) => {
                let pairs = context
                    .pairs
                    .into_iter()
                    .map(|pair| {
                        Ok(ContextPair {
                            positive: ids_to_vectors
                                .resolve_reference(
                                    lookup_collection,
                                    lookup_vector_name,
                                    pair.positive,
                                )
                                .ok_or_else(|| vector_not_found_error(lookup_vector_name))?,
                            negative: ids_to_vectors
                                .resolve_reference(
                                    lookup_collection,
                                    lookup_vector_name,
                                    pair.negative,
                                )
                                .ok_or_else(|| vector_not_found_error(lookup_vector_name))?,
                        })
                    })
                    .collect::<CollectionResult<_>>()?;

                Ok(VectorQuery::Context(ContextQuery { pairs }))
            }
            VectorQuery::NearestWithMmr(NearestWithMmr { nearest, mmr }) => {
                let nearest = ids_to_vectors
                    .resolve_reference(lookup_collection, lookup_vector_name, nearest)
                    .ok_or_else(|| vector_not_found_error(lookup_vector_name))?;

                Ok(VectorQuery::NearestWithMmr(NearestWithMmr { nearest, mmr }))
            }
            VectorQuery::NearestWithFocus(NearestWithFocus { nearest, focus }) => {
                let nearest = ids_to_vectors
                    .resolve_reference(lookup_collection, lookup_vector_name, nearest)
                    .ok_or_else(|| vector_not_found_error(lookup_vector_name))?;

                Ok(VectorQuery::NearestWithFocus(NearestWithFocus {
                    nearest,
                    focus,
                }))
            }
            VectorQuery::Feedback(FeedbackInternal {
                target,
                feedback,
                strategy,
            }) => {
                let target = ids_to_vectors
                    .resolve_reference(lookup_collection, lookup_vector_name, target)
                    .ok_or_else(|| vector_not_found_error(lookup_vector_name))?;

                let feedback = feedback
                    .into_iter()
                    .map(|FeedbackItem { vector, score }| {
                        Ok(FeedbackItem {
                            vector: ids_to_vectors
                                .resolve_reference(lookup_collection, lookup_vector_name, vector)
                                .ok_or_else(|| vector_not_found_error(lookup_vector_name))?,
                            score,
                        })
                    })
                    .collect::<CollectionResult<_>>()?;

                Ok(VectorQuery::Feedback(FeedbackInternal {
                    target,
                    feedback,
                    strategy,
                }))
            }
        }
    }

    /// Resolves the references in the RecoQuery into actual vectors.
    fn resolve_reco_reference(
        reco_query: RecoQuery<VectorInputInternal>,
        ids_to_vectors: &ReferencedVectors,
        lookup_vector_name: &VectorName,
        lookup_collection: Option<&String>,
    ) -> (Vec<VectorInternal>, Vec<VectorInternal>) {
        let positives = reco_query
            .positives
            .into_iter()
            .filter_map(|vector_input| {
                ids_to_vectors.resolve_reference(
                    lookup_collection,
                    lookup_vector_name,
                    vector_input,
                )
            })
            .collect();
        let negatives = reco_query
            .negatives
            .into_iter()
            .filter_map(|vector_input| {
                ids_to_vectors.resolve_reference(
                    lookup_collection,
                    lookup_vector_name,
                    vector_input,
                )
            })
            .collect();
        (positives, negatives)
    }
}

fn vector_not_found_error(vector_name: &VectorName) -> CollectionError {
    CollectionError::not_found(format!("Vector with name {vector_name:?} for point"))
}

impl VectorQuery<VectorInternal> {
    fn preprocess_vectors(mut self) -> Self {
        match &mut self {
            VectorQuery::Nearest(vector) => {
                vector.preprocess();
            }
            VectorQuery::RecommendAverageVector(reco) => {
                reco.positives.iter_mut().for_each(|v| v.preprocess());
                reco.negatives.iter_mut().for_each(|v| v.preprocess());
            }
            VectorQuery::RecommendBestScore(reco) => {
                reco.positives.iter_mut().for_each(|v| v.preprocess());
                reco.negatives.iter_mut().for_each(|v| v.preprocess());
            }
            VectorQuery::RecommendSumScores(reco) => {
                reco.positives.iter_mut().for_each(|v| v.preprocess());
                reco.negatives.iter_mut().for_each(|v| v.preprocess());
            }
            VectorQuery::Discover(discover) => {
                discover.target.preprocess();
                discover.pairs.iter_mut().for_each(|pair| {
                    pair.positive.preprocess();
                    pair.negative.preprocess();
                });
            }
            VectorQuery::Context(context) => {
                context.pairs.iter_mut().for_each(|pair| {
                    pair.positive.preprocess();
                    pair.negative.preprocess();
                });
            }
            VectorQuery::NearestWithMmr(NearestWithMmr { nearest, mmr: _ }) => {
                nearest.preprocess();
            }
            VectorQuery::NearestWithFocus(NearestWithFocus { nearest, focus: _ }) => {
                nearest.preprocess();
            }
            VectorQuery::Feedback(FeedbackInternal {
                target,
                feedback,
                strategy: _,
            }) => {
                target.preprocess();
                feedback
                    .iter_mut()
                    .for_each(|item| item.vector.preprocess());
            }
        }
        self
    }

    fn into_scoring_query(
        self,
        using: VectorNameBuf,
        request_limit: usize,
    ) -> CollectionResult<ScoringQuery> {
        let query_enum = match self {
            VectorQuery::Nearest(vector) => QueryEnum::Nearest(NamedQuery::new(vector, using)),
            VectorQuery::RecommendAverageVector(reco) => {
                // Get average vector
                let search_vector = avg_vector_for_recommendation(
                    reco.positives.iter().map(VectorRef::from),
                    reco.negatives.iter().map(VectorRef::from).peekable(),
                )?;
                QueryEnum::Nearest(NamedQuery::new(search_vector, using))
            }
            VectorQuery::RecommendBestScore(reco) => {
                QueryEnum::RecommendBestScore(NamedQuery::new(reco, using))
            }
            VectorQuery::RecommendSumScores(reco) => {
                QueryEnum::RecommendSumScores(NamedQuery::new(reco, using))
            }
            VectorQuery::Discover(discover) => {
                QueryEnum::Discover(NamedQuery::new(discover, using))
            }
            VectorQuery::Context(context) => QueryEnum::Context(NamedQuery::new(context, using)),
            VectorQuery::NearestWithMmr(NearestWithMmr { nearest, mmr }) => {
                let Mmr {
                    diversity,
                    candidates_limit,
                } = mmr;

                return Ok(ScoringQuery::Mmr(MmrInternal {
                    vector: nearest,
                    using,
                    lambda: OrderedFloat(diversity.map(|x| 1.0 - x).unwrap_or(DEFAULT_MMR_LAMBDA)),
                    candidates_limit: candidates_limit.unwrap_or(request_limit),
                }));
            }
            VectorQuery::NearestWithFocus(NearestWithFocus { nearest, focus }) => {
                if !matches!(nearest, VectorInternal::Dense(_)) {
                    return Err(CollectionError::bad_request(
                        "Focused search on dimensions is only supported for dense vectors.",
                    ));
                }

                let DimsFocus {
                    dims,
                    candidates_limit,
                    masked,
                    mask_from_layer,
                    verify,
                    alpha,
                } = focus;

                if mask_from_layer.is_some() && !masked {
                    return Err(CollectionError::bad_request(
                        "mask_from_layer requires focus.masked = true.",
                    ));
                }

                if verify && !masked {
                    return Err(CollectionError::bad_request(
                        "verify requires focus.masked = true.",
                    ));
                }

                if alpha.is_some() && !masked {
                    return Err(CollectionError::bad_request(
                        "alpha requires focus.masked = true.",
                    ));
                }

                if let Some(alpha) = alpha
                    && !(0.0..=1.0).contains(&alpha)
                {
                    return Err(CollectionError::bad_request("alpha must be in [0, 1]."));
                }

                // Masked focus computes the reduced distance inside the index traversal,
                // so it is a plain nearest-neighbor scoring query (no preselect + rescore).
                // When verify=true, segment postprocess re-scores the returned top-k with
                // full-vector distance so returned scores are exact full-space scores.
                // When alpha is set, masked layers blend full + focus scores (Option 4).
                if masked {
                    let mask_from_layer = mask_from_layer.map(|layer| layer as usize);
                    // mask_from_layer = 0 → no masking → identical to plain nearest.
                    // verify / alpha are no-ops here (scores are already full-distance).
                    if mask_from_layer == Some(0) {
                        return Ok(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery::new(
                            nearest, using,
                        ))));
                    }
                    return Ok(ScoringQuery::Vector(QueryEnum::NearestMasked(
                        NamedQuery::new(nearest, using),
                        dims,
                        mask_from_layer,
                        verify,
                        alpha.map(ordered_float::OrderedFloat),
                    )));
                }

                return Ok(ScoringQuery::DimsFocus(DimsFocusInternal {
                    vector: nearest,
                    using,
                    dims,
                    candidates_limit: candidates_limit.unwrap_or(request_limit),
                }));
            }
            VectorQuery::Feedback(FeedbackInternal {
                target,
                feedback,
                strategy,
            }) => match strategy {
                FeedbackStrategy::Naive { a, b, c } => QueryEnum::FeedbackNaive(NamedQuery::new(
                    segment::vector_storage::query::NaiveFeedbackQuery {
                        target,
                        feedback,
                        coefficients: NaiveFeedbackCoefficients {
                            a: a.into(),
                            b: b.into(),
                            c: c.into(),
                        },
                    },
                    using,
                )),
            },
        };

        Ok(ScoringQuery::Vector(query_enum))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectionPrefetch {
    pub prefetch: Vec<CollectionPrefetch>,
    pub query: Option<Query>,
    pub using: VectorNameBuf,
    pub filter: Option<Filter>,
    pub score_threshold: Option<OrderedFloat<ScoreType>>,
    pub limit: usize,
    /// Search params for when there is no prefetch
    pub params: Option<SearchParams>,
    pub lookup_from: Option<LookupLocation>,
}

/// Exclude the referenced ids by editing the filter.
fn exclude_referenced_ids(ids: Vec<ExtendedPointId>, filter: Option<Filter>) -> Option<Filter> {
    let ids: AHashSet<_> = ids.into_iter().collect();

    if ids.is_empty() {
        return filter;
    }

    let id_filter = Filter::new_must_not(Condition::HasId(HasIdCondition::from(ids)));
    Some(id_filter.merge_owned(filter.unwrap_or_default()))
}

impl CollectionPrefetch {
    fn get_lookup_collection(&self) -> Option<&String> {
        self.lookup_from.as_ref().map(|x| &x.collection)
    }

    fn get_lookup_vector_name(&self) -> VectorNameBuf {
        self.lookup_from
            .as_ref()
            .and_then(|lookup_from| lookup_from.vector.as_ref())
            .unwrap_or(&self.using)
            .to_owned()
    }

    pub fn get_referenced_point_ids_on_collection(&self, collection: &str) -> Vec<PointIdType> {
        let mut refs = Vec::new();

        let mut lookup_other_collection = false;
        if let Some(lookup_collection) = self.get_lookup_collection() {
            lookup_other_collection = lookup_collection != collection
        };

        if !lookup_other_collection && let Some(Query::Vector(vector_query)) = &self.query {
            if let VectorQuery::Nearest(VectorInputInternal::Id(id)) = vector_query {
                refs.push(*id);
            }
            refs.extend(vector_query.get_referenced_ids())
        };

        for prefetch in &self.prefetch {
            refs.extend(prefetch.get_referenced_point_ids_on_collection(collection))
        }

        refs
    }

    fn try_into_shard_prefetch(
        self,
        ids_to_vectors: &ReferencedVectors,
    ) -> CollectionResult<ShardPrefetch> {
        CollectionQueryRequest::validation(
            &self.query,
            &self.using,
            &self.prefetch,
            self.score_threshold.map(OrderedFloat::into_inner),
        )?;

        let lookup_vector_name = self.get_lookup_vector_name();
        let lookup_collection = self.get_lookup_collection().cloned();
        let using = self.using.clone();

        let query = self
            .query
            .map(|query| {
                query.try_into_scoring_query(
                    ids_to_vectors,
                    &lookup_vector_name,
                    lookup_collection.as_ref(),
                    using,
                    self.limit,
                )
            })
            .transpose()?;

        let prefetches = self
            .prefetch
            .into_iter()
            .map(|prefetch| prefetch.try_into_shard_prefetch(ids_to_vectors))
            .try_collect()?;

        Ok(ShardPrefetch {
            prefetches,
            query,
            filter: self.filter,
            score_threshold: self.score_threshold,
            limit: self.limit,
            params: self.params,
        })
    }

    pub fn flatten_resolver_requests(&self) -> Vec<CollectionQueryResolveRequest> {
        let mut inner_queries = vec![];
        // resolve ids for root query
        let referenced_ids = self
            .query
            .as_ref()
            .map(Query::get_referenced_ids)
            .unwrap_or_default();

        if !referenced_ids.is_empty() {
            let resolve_root = CollectionQueryResolveRequest {
                referenced_ids,
                lookup_from: self.lookup_from.clone(),
                using: self.using.clone(),
            };
            inner_queries.push(resolve_root);
        }

        // recurse on prefetches
        for prefetch in &self.prefetch {
            for flatten in prefetch.flatten_resolver_requests() {
                inner_queries.push(flatten);
            }
        }
        inner_queries
    }
}

impl CollectionQueryRequest {
    fn get_lookup_collection(&self) -> Option<&String> {
        self.lookup_from.as_ref().map(|x| &x.collection)
    }

    fn get_lookup_vector_name(&self) -> VectorNameBuf {
        self.lookup_from
            .as_ref()
            .and_then(|lookup_from| lookup_from.vector.as_ref())
            .unwrap_or(&self.using)
            .to_owned()
    }

    fn get_referenced_point_ids_on_collection(&self, collection: &str) -> Vec<PointIdType> {
        let mut refs = Vec::new();

        let mut lookup_other_collection = false;
        if let Some(lookup_collection) = self.get_lookup_collection() {
            lookup_other_collection = lookup_collection != collection
        };

        if !lookup_other_collection && let Some(Query::Vector(vector_query)) = &self.query {
            if let VectorQuery::Nearest(VectorInputInternal::Id(id)) = vector_query {
                refs.push(*id);
            }
            refs.extend(vector_query.get_referenced_ids())
        };

        for prefetch in &self.prefetch {
            refs.extend(prefetch.get_referenced_point_ids_on_collection(collection))
        }

        refs
    }

    /// Substitutes all the point ids in the request with the actual vectors, as well as editing filters so that ids are not included in the response.
    pub fn try_into_shard_request(
        self,
        collection_name: &str,
        ids_to_vectors: &ReferencedVectors,
    ) -> CollectionResult<ShardQueryRequest> {
        Self::validation(
            &self.query,
            &self.using,
            &self.prefetch,
            self.score_threshold,
        )?;

        let mut offset = self.offset;
        if matches!(self.query, Some(Query::Sample(SampleInternal::Random)))
            && self.prefetch.is_empty()
        {
            // Shortcut: Ignore offset with random query, since output is not stable.
            offset = 0;
        }

        let query_lookup_collection = self.get_lookup_collection().cloned();
        let query_lookup_vector_name = self.get_lookup_vector_name();
        let using = self.using.clone();

        // Edit filter to exclude all referenced point ids (root and nested) on the searched collection
        // We do not want to exclude vector ids from different collection via lookup_from.
        let referenced_point_ids = self.get_referenced_point_ids_on_collection(collection_name);

        let filter = exclude_referenced_ids(referenced_point_ids, self.filter);

        if self.dims_explained.is_some() {
            Self::validate_dims_explained_query_type(self.query.as_ref())?;
        }

        let query = self
            .query
            .map(|query| {
                query.try_into_scoring_query(
                    ids_to_vectors,
                    &query_lookup_vector_name,
                    query_lookup_collection.as_ref(),
                    using,
                    self.limit,
                )
            })
            .transpose()?;

        let prefetches = self
            .prefetch
            .into_iter()
            .map(|prefetch| prefetch.try_into_shard_prefetch(ids_to_vectors))
            .try_collect()?;

        if self.dims_explained.is_some() {
            Self::validate_dims_explained_support(query.as_ref())?;
        }

        Ok(ShardQueryRequest {
            prefetches,
            query,
            filter,
            score_threshold: self.score_threshold.map(OrderedFloat),
            limit: self.limit,
            offset,
            params: self.params,
            with_vector: self.with_vector,
            with_payload: self.with_payload,
            dims_explained: self.dims_explained,
        })
    }

    /// Check that the request query type supports per-dimension score explanations.
    ///
    /// This is validated before query conversion, so recommend/discover/etc. are rejected
    /// even when they would be lowered to a nearest search internally.
    fn validate_dims_explained_query_type(query: Option<&Query>) -> CollectionResult<()> {
        let supported = matches!(
            query,
            Some(Query::Vector(VectorQuery::Nearest(_)))
                | Some(Query::Vector(VectorQuery::NearestWithFocus(_)))
        );

        if !supported {
            return Err(CollectionError::bad_request(
                "with_dims_explained is only supported for nearest queries against dense vectors.",
            ));
        }

        Ok(())
    }

    /// Check that the resolved query supports per-dimension score explanations.
    ///
    /// Score decomposition per dimension is only defined for nearest neighbor scoring
    /// (optionally focused on a subset of dimensions) against dense vectors.
    fn validate_dims_explained_support(query: Option<&ScoringQuery>) -> CollectionResult<()> {
        // Masked focus search scores inside the index traversal and does not retrieve the
        // full vectors needed to decompose the score, so explanations are not available.
        if let Some(ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, _, _, _))) = query {
            return Err(CollectionError::bad_request(
                "with_dims_explained cannot be combined with masked focus search.",
            ));
        }

        let supported = match query {
            Some(ScoringQuery::Vector(QueryEnum::Nearest(nearest))) => {
                matches!(nearest.query, VectorInternal::Dense(_))
            }
            Some(ScoringQuery::DimsFocus(focus)) => {
                matches!(focus.vector, VectorInternal::Dense(_))
            }
            _ => false,
        };

        if !supported {
            return Err(CollectionError::bad_request(
                "with_dims_explained is only supported for nearest queries against dense vectors.",
            ));
        }

        Ok(())
    }

    pub fn validation(
        query: &Option<Query>,
        using: &VectorNameBuf,
        prefetch: &[CollectionPrefetch],
        score_threshold: Option<ScoreType>,
    ) -> CollectionResult<()> {
        // Check no prefetches without a query
        if !prefetch.is_empty() && query.is_none() {
            return Err(CollectionError::bad_request(
                "A query is needed to merge the prefetches. Can't have prefetches without defining a query.",
            ));
        }

        // Check no score_threshold without a vector query
        if score_threshold.is_some() {
            match query {
                Some(Query::OrderBy(_)) => {
                    return Err(CollectionError::bad_request(
                        "Can't use score_threshold with an order_by query.",
                    ));
                }
                None => {
                    return Err(CollectionError::bad_request(
                        "A query is needed to use the score_threshold. Can't have score_threshold without defining a query.",
                    ));
                }
                _ => {}
            }
        }

        // Check that fusion queries are not combined with a using vector name
        if let Some(Query::Fusion(_)) = query
            && using != DEFAULT_VECTOR_NAME
        {
            return Err(CollectionError::bad_request(
                "Fusion queries cannot be combined with the 'using' field.",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod mask_from_layer_tests {
    use super::*;

    fn dense_focus_query(
        masked: bool,
        mask_from_layer: Option<u32>,
    ) -> VectorQuery<VectorInternal> {
        VectorQuery::NearestWithFocus(NearestWithFocus {
            nearest: VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0]),
            focus: DimsFocus {
                dims: vec![0, 2],
                candidates_limit: None,
                masked,
                mask_from_layer,
                verify: false,
                alpha: None,
            },
        })
    }

    #[test]
    fn mask_from_layer_requires_masked() {
        let err = dense_focus_query(false, Some(1))
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap_err();
        assert!(err.to_string().contains("mask_from_layer"));
    }

    #[test]
    fn mask_from_layer_zero_becomes_plain_nearest() {
        let scored = dense_focus_query(true, Some(0))
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        assert!(matches!(
            scored,
            ScoringQuery::Vector(QueryEnum::Nearest(_))
        ));
    }

    #[test]
    fn omitted_mask_from_layer_is_legacy_full_mask() {
        let scored = dense_focus_query(true, None)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, None, false, None)) => {}
            other => panic!("expected NearestMasked(..., None), got {other:?}"),
        }
    }

    #[test]
    fn mask_from_layer_hybrid_is_threaded() {
        let scored = dense_focus_query(true, Some(2))
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, Some(2), false, None)) => {}
            other => panic!("expected NearestMasked(..., Some(2)), got {other:?}"),
        }
    }

    #[test]
    fn masked_rejects_non_dense() {
        let query = VectorQuery::NearestWithFocus(NearestWithFocus {
            nearest: VectorInternal::MultiDense(
                segment::data_types::vectors::MultiDenseVectorInternal::new(
                    vec![1.0, 2.0, 3.0, 4.0],
                    2,
                ),
            ),
            focus: DimsFocus {
                dims: vec![0],
                candidates_limit: None,
                masked: true,
                mask_from_layer: Some(1),
                verify: false,
                alpha: None,
            },
        });
        let err = query
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap_err();
        assert!(err.to_string().contains("dense"));
    }
}

#[cfg(test)]
mod verify_tests {
    use super::*;

    fn dense_focus_query(masked: bool, verify: bool) -> VectorQuery<VectorInternal> {
        VectorQuery::NearestWithFocus(NearestWithFocus {
            nearest: VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0]),
            focus: DimsFocus {
                dims: vec![0, 2],
                candidates_limit: None,
                masked,
                mask_from_layer: None,
                verify,
                alpha: None,
            },
        })
    }

    #[test]
    fn verify_requires_masked() {
        let err = dense_focus_query(false, true)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap_err();
        assert!(err.to_string().contains("verify"));
        assert!(err.to_string().contains("masked"));
    }

    #[test]
    fn verify_false_is_legacy_masked() {
        let scored = dense_focus_query(true, false)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, None, false, None)) => {}
            other => panic!("expected NearestMasked(..., false), got {other:?}"),
        }
    }

    #[test]
    fn verify_true_is_threaded() {
        let scored = dense_focus_query(true, true)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, None, true, None)) => {}
            other => panic!("expected NearestMasked(..., true), got {other:?}"),
        }
    }

    #[test]
    fn verify_with_hybrid_is_threaded() {
        let scored = VectorQuery::NearestWithFocus(NearestWithFocus {
            nearest: VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0]),
            focus: DimsFocus {
                dims: vec![0, 2],
                candidates_limit: None,
                masked: true,
                mask_from_layer: Some(1),
                verify: true,
                alpha: None,
            },
        })
        .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
        .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, Some(1), true, None)) => {}
            other => panic!("expected NearestMasked(..., Some(1), true), got {other:?}"),
        }
    }
}

#[cfg(test)]
mod alpha_tests {
    use super::*;

    fn dense_focus_query(
        masked: bool,
        alpha: Option<f32>,
        mask_from_layer: Option<u32>,
        verify: bool,
    ) -> VectorQuery<VectorInternal> {
        VectorQuery::NearestWithFocus(NearestWithFocus {
            nearest: VectorInternal::Dense(vec![1.0, 2.0, 3.0, 4.0]),
            focus: DimsFocus {
                dims: vec![0, 2],
                candidates_limit: None,
                masked,
                mask_from_layer,
                verify,
                alpha,
            },
        })
    }

    #[test]
    fn alpha_requires_masked() {
        let err = dense_focus_query(false, Some(0.5), None, false)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap_err();
        assert!(err.to_string().contains("alpha"));
        assert!(err.to_string().contains("masked"));
    }

    #[test]
    fn alpha_out_of_range_is_rejected() {
        for bad in [-0.1_f32, 1.1] {
            let err = dense_focus_query(true, Some(bad), None, false)
                .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
                .unwrap_err();
            assert!(
                err.to_string().contains("alpha") && err.to_string().contains("[0, 1]"),
                "unexpected error for {bad}: {err}"
            );
        }
    }

    #[test]
    fn alpha_is_threaded() {
        let scored = dense_focus_query(true, Some(0.25), None, false)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, None, false, Some(a))) => {
                assert!((a.into_inner() - 0.25).abs() < 1e-6);
            }
            other => panic!("expected NearestMasked(..., Some(0.25)), got {other:?}"),
        }
    }

    #[test]
    fn alpha_with_hybrid_and_verify_is_threaded() {
        let scored = dense_focus_query(true, Some(0.5), Some(1), true)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, Some(1), true, Some(a))) => {
                assert!((a.into_inner() - 0.5).abs() < 1e-6);
            }
            other => panic!("expected hybrid+verify+alpha, got {other:?}"),
        }
    }

    #[test]
    fn omitted_alpha_is_none() {
        let scored = dense_focus_query(true, None, None, false)
            .into_scoring_query(DEFAULT_VECTOR_NAME.to_owned(), 10)
            .unwrap();
        match scored {
            ScoringQuery::Vector(QueryEnum::NearestMasked(_, _, None, false, None)) => {}
            other => panic!("expected NearestMasked(..., None alpha), got {other:?}"),
        }
    }
}
