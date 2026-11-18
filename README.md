# XQdrant

**Explainable, focus-aware vector search. a research fork of [Qdrant](https://github.com/qdrant/qdrant).**

Vector databases return a single similarity score and no reason for it. XQdrant
adds two capabilities to Qdrant's Query API and measures what they cost:

1. **Attribution**: `with_dims_explained`: each hit comes back with its exact
   top-*m* per-dimension score breakdown, computed in the engine.
2. **Focused search**: `focus.dims`: score (and traverse HNSW) using only a
   chosen subset of dimensions, with a hybrid mode that keeps full-distance
   routing on the upper layers (`mask_from_layer`) and an optional full-distance
   re-check (`verify`).

> Status: research prototype accompanying a paper submitted to EDBT 2027
> (*XQdrant: In-Database Attribution and Masked-Distance Subspace Search over HNSW*).
> Not production-ready. Feedback from Qdrant engineers is very welcome.

> **Branches:** `xq/main` has attribution (`with_dims_explained`).
> The full implementation (focused search, hybrid cutoff, verify, kernels, benches)
> is on [`option2-hybrid-layer-cutoff`](https://github.com/Moetez-Fradi/XQdrant/tree/option2-hybrid-layer-cutoff).
> Other branches are mirrored from upstream Qdrant.

---

## Try it

Explain a normal search:

```json
POST /collections/{name}/points/query
{
  "query": { "nearest": [0.1, 0.2, "..."] },
  "with_dims_explained": { "top": 5 },
  "limit": 10
}
```

Focused search (cannot be combined with `with_dims_explained`):

```json
POST /collections/{name}/points/query
{
  "query": {
    "nearest": [0.1, 0.2, "..."],
    "focus": { "dims": [0, 1, 2, 3], "masked": true, "mask_from_layer": 1, "verify": true }
  },
  "limit": 10
}
```

| Field | Meaning |
|---|---|
| `with_dims_explained: { "top": m }` | Attach the *m* dimensions contributing most to each hit's score |
| `focus.dims` | Dimensions that define the focus subspace |
| `focus.masked` | Use the masked (focus-only) distance during HNSW traversal |
| `focus.mask_from_layer` | Layer *L* uses the masked distance when *L* < cutoff; `0` = plain search |
| `focus.verify` | Re-score the returned candidates with the full distance |

Build and run like upstream Qdrant (`cargo build --release`, binary
`./target/release/qdrant`). On the full branch, set
`XQDRANT_MASKED_KERNEL=gather` or `repack` (default) to pick the masked kernel.

---

## Key findings

| Finding | Result |
|---|---|
| **Attribution is cheap** | +0.3 ms in-engine, independent of *m*; ~1.8× faster end to end than extracting it client-side (D = 768, 1536). Ranking unchanged. |
| **The ceiling is the data, not the index** | Neighbors in a subspace are mostly *not* neighbors in the full space. On SIFT1M at 10% of dims, masked search matches the exact subspace/full-space overlap (0.012 vs 0.013). |
| **Masked HNSW finds subspace neighbors** | Recall@10 ≥ 0.91 vs subspace ground truth when ≥ half the dims are in focus; 0.77 at a quarter, ~0.4 at a tenth. |
| **No latency win yet** | Masked search runs at 0.67–0.85× plain speed. A masked distance over 25% of dims costs ~2× a full contiguous SIMD distance — per-distance cost, not traversal, is the bottleneck. |
| **Verify helps where data allows** | Overfetch + full rescoring lifts full-space recall 0.48 → 0.91 at 75% of dims, for < 0.3 ms. |

Negative results (weighted full/masked blend never beats pure masking) are
reported in the paper too.

---

## Where the code lives

| Area | Path |
|---|---|
| Attribution | `lib/segment/src/common/dims_explained.rs` |
| REST / gRPC schema | `lib/api/src/rest/schema.rs`, `lib/api/src/grpc/proto/points.proto` |
| Query routing | `lib/collection/.../collection_query.rs` |
| Masked scorer + kernels | `lib/segment/src/vector_storage/query_scorer/masked_*.rs` |
| Per-layer scorer switch | `lib/segment/.../point_scorer.rs` (`FilteredScorer`), `graph_layers.rs` (`set_layer`) |
| Verify rescore | `lib/segment/.../vector_index_search_common.rs` (`postprocess_search_result`) |
| Kernel bench | `lib/segment/benches/masked_kernel_bench.rs` |

Benchmarks, harness and the experiment index (each claim → script):
[Moetez-Fradi/XAI › `DB_systems/`](https://github.com/Moetez-Fradi/XAI/tree/EDBT27/DB_systems)
(paper source in `DB_systems/deliverable/`).

---

## Open questions

- Can a masked distance ever beat contiguous SIMD, e.g. with a dims-major
  layout or per-query repacked segments?
- **Mixed precision per layer:** the `set_layer` switch could pair cheap
  quantization (e.g. TurboQuant 1-bit) on upper layers with higher precision on layer 0.
- Is per-dimension attribution useful to real users, or should it target
  named-vector / multivector "aspects" instead?
- Which piece, if any, would be worth upstreaming, and in what shape?
