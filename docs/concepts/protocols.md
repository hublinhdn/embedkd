# Retrieval protocols

## Gallery-query

Evaluation ranks every **query** image against a **gallery** by cosine
similarity of L2-normalised embeddings and reports:

- **mAP**: mean average precision over queries (a query's relevant items are
  the gallery images with the same label).
- **R@k**: fraction of queries whose top-k contains at least one relevant item.
- **retention**: student mAP / teacher mAP, when the teacher is available.

Queries with no relevant gallery item are skipped and counted in
`num_queries`.

## Split semantics

- `split.mode: auto` (closed-set): every class contributes images to train,
  gallery and query. Generated once, frozen to a CSV in the data root,
  re-read afterwards.
- Open-set (used by the built-in CUB / Cars / SOP adapters): train classes
  and evaluation classes are disjoint; each evaluation class's images are
  split into disjoint gallery and query halves deterministically. This
  measures whether the embedding generalises to unseen classes.

!!! note "Documented divergence"
    Some papers evaluate CUB/Cars/SOP leave-one-out (each test image queries
    all others). EmbedKD uses disjoint gallery/query halves for a clean
    protocol; numbers are therefore comparable within EmbedKD runs, and the
    frozen split files ship with the reproduction demos.

## Cross-domain zero-shot

`data.protocol: cross_domain` adds a `data.target` block; after training on
the source, the model is evaluated on the target's gallery-query pair with no
fine-tuning (`embedkd eval --target`). This is the deployment question: does
the distilled embedding survive a domain shift?
