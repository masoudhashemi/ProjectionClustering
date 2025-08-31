import argparse
import csv
import json
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from direction_calculator import DirectionCalculator
from embedding_processor import EmbeddingProcessor


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Projection-based similarity using a perspective direction")
    p.add_argument("--texts-file", type=str, default=None, help="Path to file with input texts (one per line)")
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")

    # Perspective (optional). If provided => use projection. If omitted => cosine-based clustering.
    p.add_argument("--perspective", type=str, default=None, help="Perspective text to define the direction")
    p.add_argument(
        "--reference",
        action="append",
        default=None,
        help="Reference text(s) to contrast with the perspective (use multiple times)",
    )
    p.add_argument("--reference-file", type=str, default=None, help="Path to file with reference texts (one per line)")
    # References are optional; if none provided, perspective alone defines the direction.

    # Direction quality options
    p.add_argument("--centroid-method", choices=["mean", "trimmed", "huber"], default="mean", help="Centroid estimator for perspective/reference sets")
    p.add_argument("--trim-proportion", type=float, default=0.1, help="Trim proportion for trimmed centroid (each side, 0-0.49)")
    p.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta for robust centroid")
    p.add_argument("--ref-weight", type=float, default=1.0, help="Scale for reference centroid in direction: dir = P - ref_weight*R")

    p.add_argument(
        "--cluster-mode",
        choices=["none", "kmeans"],
        default="none",
        help="Cluster scores with KMeans. With perspective: cluster 1D projection scores. Without perspective: cluster normalized embeddings (cosine-like).",
    )
    p.add_argument("--kmeans-k", type=int, default=3, help="K for KMeans when clustering scores")

    # Plotting
    p.add_argument("--no-plot", action="store_true", help="Disable plots (useful for headless runs)")
    p.add_argument("--top-k", type=int, default=20, help="Show top-K results in console output")

    # Output saving
    p.add_argument("--out-csv", type=str, default=None, help="Path to save scores/clusters as CSV")
    p.add_argument("--out-json", type=str, default=None, help="Path to save scores/clusters as JSON")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    # 1) Load texts
    if args.texts_file:
        texts = read_lines(args.texts_file)
    else:
        # Generic but meaningful sample corpus for quick demos
        texts = [
            "Guide to planning a small project.",
            "Steps for brewing a cup of coffee.",
            "Short tutorial on organizing digital files.",
            "Overview of budgeting for monthly expenses.",
            "Tips for writing clear emails at work.",
            "Checklist for preparing a weekend trip.",
            "Basics of stretching for daily exercise.",
            "Notes from a team meeting about timelines.",
            "Summary of a book chapter on creativity.",
            "Simple recipe for a vegetable stir-fry.",
            "Advice on keeping indoor plants healthy.",
            "Introduction to sorting data in spreadsheets.",
        ]

    # 2) Embeddings
    embedder = EmbeddingProcessor(model_name=args.model)
    embeddings = embedder.encode(texts)
    embeddings_dict = {text: emb for text, emb in zip(texts, embeddings)}
    # 3) Perspective / scoring
    perspective_text = args.perspective
    info_lines: List[str] = []
    scores = None
    reference_texts: Optional[List[str]] = None
    if perspective_text:
        perspective_emb = embedder.encode([perspective_text])[0]
        embeddings_dict[perspective_text] = perspective_emb

        collected_refs: List[str] = []
        if args.reference:
            collected_refs.extend(args.reference)
        if args.reference_file:
            collected_refs.extend(read_lines(args.reference_file))

        # Use references only if explicitly provided.
        if collected_refs:
            reference_texts = collected_refs

        # Ensure explicit references have embeddings
        if collected_refs:
            to_encode = [r for r in collected_refs if r not in embeddings_dict]
            if to_encode:
                ref_embs = embedder.encode(to_encode)
                for r, e in zip(to_encode, ref_embs):
                    embeddings_dict[r] = e

        direction = DirectionCalculator.compute_perspective_direction(
            perspective_texts=[perspective_text],
            embeddings_dict=embeddings_dict,
            reference_texts=reference_texts,
            centroid_method=args.centroid_method,
            trim_proportion=args.trim_proportion,
            huber_delta=args.huber_delta,
            ref_weight=args.ref_weight,
        )
        if direction is None:
            raise RuntimeError("Failed to compute perspective direction.")
        scores = np.dot(embeddings, direction)
        info_lines.append("Mode: projection (perspective provided)")
        if reference_texts is not None:
            info_lines.append(f"Reference count: {len(reference_texts)} (used only to define direction)")
    else:
        # No perspective provided: we'll support cosine-based clustering only (no ranking plot).
        info_lines.append("Mode: cosine clustering (no perspective)")

    ranking = []
    if scores is not None:
        ranking = sorted(zip(texts, scores.tolist()), key=lambda x: x[1], reverse=True)

    # 5) Console output (top-K)
    top_k = max(1, min(args.top_k, len(ranking)))
    print("Perspective:", perspective_text if perspective_text else "<none>")
    for line in info_lines:
        print(line)
    if scores is not None:
        print("\nTop aligned texts:")
        for i, (t, s) in enumerate(ranking[:top_k], start=1):
            print(f"{i:2d}. {s:+.4f}  | {t}")

        print("\nMost opposed texts:")
        for i, (t, s) in enumerate(reversed(ranking[-top_k:]), start=1):
            print(f"{i:2d}. {s:+.4f}  | {t}")

    # 6) Optional plot
    if not args.no_plot and scores is not None:
        plt.figure(figsize=(12, 5))
        idx = np.arange(len(texts))
        plt.bar(idx, scores)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.title("Scores vs. Perspective")
        plt.xlabel("Text Index")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()

    # 7) Optional clustering on scores
    if args.cluster_mode == "kmeans":
        k = max(1, min(args.kmeans_k, len(texts)))
        km = KMeans(n_clusters=k, random_state=42)
        if scores is not None:
            labels = km.fit_predict(scores.reshape(-1, 1))
            centers = km.cluster_centers_.ravel()

            print("\nClusters (KMeans on projection scores):")
            for i in range(k):
                count = int(np.sum(labels == i))
                center = float(centers[i])
                print(f"cluster {i}: center={center:+.4f}, count={count}")

            ordered = sorted(zip(texts, scores.tolist(), labels.tolist()), key=lambda x: (x[2], -x[1]))
            for t, s, lbl in ordered:
                print(f"cluster {int(lbl)} | {s:+.4f} | {t}")
        else:
            # No perspective: cluster normalized embeddings (cosine-like)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            e_unit = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)
            labels = km.fit_predict(e_unit)
            print("\nClusters (KMeans on normalized embeddings - cosine-like):")
            for i in range(k):
                count = int(np.sum(labels == i))
                print(f"cluster {i}: count={count}")
            for t, lbl in sorted(zip(texts, labels.tolist()), key=lambda x: x[1]):
                print(f"cluster {int(lbl)} | {t}")

    # 8) Save outputs if requested
    if args.out_csv or args.out_json:
        # Build rows
        rows = []
        n = len(texts)
        lbls = locals().get('labels', None)
        lbl_list = lbls.tolist() if lbls is not None else [None] * n
        score_list = scores.tolist() if scores is not None else [None] * n
        for idx, t in enumerate(texts):
            row = {
                "index": idx,
                "text": t,
                "score": score_list[idx],
                "cluster": int(lbl_list[idx]) if lbl_list[idx] is not None else None,
            }
            rows.append(row)

        meta = {
            "perspective": perspective_text if perspective_text else None,
            "reference_count": len(reference_texts) if reference_texts is not None else 0,
            "centroid_method": args.centroid_method,
            "trim_proportion": args.trim_proportion,
            "huber_delta": args.huber_delta,
            "ref_weight": args.ref_weight,
            "cluster_mode": args.cluster_mode,
            "kmeans_k": args.kmeans_k if args.cluster_mode == "kmeans" else None,
        }

        if args.out_csv:
            with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "text", "score", "cluster"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"\nSaved CSV: {args.out_csv}")

        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
