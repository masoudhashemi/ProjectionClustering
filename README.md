# ProjectionClustering (Perspective Projection)

## Quickstart

- Install dependencies:

  ```bash
  pip install sentence-transformers matplotlib numpy scikit-learn
  ```

- Run the demo with a perspective:

  ```bash
  python main.py --perspective "your perspective text here"
  ```

- Use your own texts (one per line):

  ```bash
  python main.py --texts-file path/to/texts.txt --perspective "your perspective text here"
  ```

This project uses projection-based similarity when a perspective is provided, and falls back to a naive cosine-based clustering baseline when it is not:

- Encode texts and a perspective with SentenceTransformers.
- Compute a perspective direction (optionally contrasted against a reference set).
- With a perspective: compute a perspective direction (optionally contrasted against a reference set) and project texts onto it to score alignment/opposition.
- Without a perspective: cluster normalized embeddings directly (cosine-like) using KMeans.

## CLI Usage (Perspective Mode)

You can run the demo (uses a small built-in list of texts) or provide your own corpus. The output lists top aligned and opposed texts by their projection score, and can optionally plot the scores.

- Built-in demo (projection with perspective):
  `python main.py --perspective "your perspective text here"`

- Provide your own corpus (one text per line):
  `python main.py --texts-file path/to/texts.txt --perspective "your perspective text here"`

- Add explicit references (multiple `--reference` allowed) or from file:
  `python main.py --texts-file texts.txt --perspective "your perspective" --reference "reference text 1" --reference "reference text 2" --reference-file refs.txt`

- Optional extras:
  - Disable plots: `--no-plot`
  - Limit console output to top-K: `--top-k 20`
  - Switch embedding model: `--model all-mpnet-base-v2`
  - Cluster with KMeans: `--cluster-mode kmeans --kmeans-k 3`
  - Direction quality: `--centroid-method {mean,trimmed,huber}` with `--trim-proportion 0.1` or `--huber-delta 1.0`
  - Reference weighting: `--ref-weight 1.0` scales the reference centroid in `P - w*R`
  - Save outputs: `--out-csv results.csv` and/or `--out-json results.json`

Notes:

- Perspective direction is the normalized vector from the centroid of the reference set to the centroid of the perspective text(s). References are only used to define this direction; they do not change any other behavior. If no references are provided, the perspective alone defines the direction (cosine-like).
- Projections near +1 align with the perspective; near -1 oppose it (sign meaningful when a reference is used).

## Options

- `--texts-file PATH`: Input corpus file, one text per line. If omitted, uses a small built-in demo set.
- `--model NAME`: SentenceTransformers model to load (default: `all-MiniLM-L6-v2`).
- `--perspective TEXT`: Perspective text that defines the direction (optional). If omitted, only clustering on normalized embeddings is available.
- `--reference TEXT`: Reference text(s) to contrast with the perspective; may be used multiple times.
- `--reference-file PATH`: File with reference texts, one per line.
- `--no-plot`: Disable plotting (useful for headless runs / CI).
- `--top-k INT`: Number of rows to print for top aligned/opposed (default: `20`).
- `--cluster-mode {none,kmeans}`: With perspective: cluster 1D projection scores. Without perspective: cluster normalized embeddings (cosine-like). Default: `none`.
- `--kmeans-k INT`: Number of clusters for KMeans on scores (default: `3`).
- `--centroid-method {mean,trimmed,huber}`: Estimator for centroids of perspective/reference sets.
- `--trim-proportion FLOAT`: Trim proportion per side used in trimmed centroid (default: `0.1`).
- `--huber-delta FLOAT`: Huber delta for robust centroid (default: `1.0`).
- `--ref-weight FLOAT`: Scale reference centroid: direction = P - ref_weight * R (default: `1.0`).
- `--out-csv PATH`: Save per-text index, text, score (if any), and cluster label (if any).
- `--out-json PATH`: Save results and metadata to JSON.

## Examples

- Cosine-like perspective scoring (no references provided):

  ```bash
  python main.py --texts-file data/texts.txt --perspective "your perspective text here"
  ```

- Projection scoring with references and KMeans clustering:

  ```bash
  python main.py --texts-file data/texts.txt \
    --perspective "your perspective" \
    --reference "reference text 1" --reference-file refs.txt \
    --cluster-mode kmeans --kmeans-k 3
  ```

- No perspective (cosine-like) clustering:

  ```bash
  python main.py --texts-file data/texts.txt \
    --cluster-mode kmeans --kmeans-k 4
  ```
