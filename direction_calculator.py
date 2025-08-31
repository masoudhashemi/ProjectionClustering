import numpy as np


class DirectionCalculator:
    @staticmethod
    def normalize_embeddings(embeddings_dict):
        """
        Normalize all embeddings to unit length.
        """
        normalized_dict = {}
        for text, emb in embeddings_dict.items():
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized_dict[text] = emb / norm
            else:
                normalized_dict[text] = emb
        return normalized_dict

    @staticmethod
    def compute_group_centroid(texts, embeddings_dict, method="mean", trim_proportion=0.1, huber_delta=1.0):
        """
        Compute and return a robust centroid of a list of texts using embeddings.

        method:
          - "mean": simple arithmetic mean.
          - "trimmed": per-dimension trimmed mean with proportion `trim_proportion` on each side.
          - "huber": Huber M-estimator with parameter `huber_delta` (iterative reweighting).
        """
        if not texts:
            return None
        embeddings = np.asarray([embeddings_dict[text] for text in texts])

        if method == "mean" or embeddings.shape[0] == 1:
            centroid = np.mean(embeddings, axis=0)
        elif method == "trimmed":
            n = embeddings.shape[0]
            p = max(0.0, min(0.49, float(trim_proportion)))
            k = int(p * n)
            if 2 * k >= n:
                centroid = np.mean(embeddings, axis=0)
            else:
                # Sort each column and trim k from both ends
                sorted_cols = np.sort(embeddings, axis=0)
                centroid = np.mean(sorted_cols[k:n - k, :], axis=0)
        elif method == "huber":
            # Iteratively reweighted least squares with Huber weights on vector residual norm
            mu = np.mean(embeddings, axis=0)
            delta = float(huber_delta)
            for _ in range(5):
                residuals = embeddings - mu
                norms = np.linalg.norm(residuals, axis=1)
                # Avoid divide-by-zero; weight = 1 when norm <= delta, else delta/norm
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = np.where(norms <= delta, 1.0, delta / np.maximum(norms, 1e-12))
                # Weighted mean
                weights = w[:, None]
                mu = np.sum(weights * embeddings, axis=0) / (np.sum(w) + 1e-12)
            centroid = mu
        else:
            centroid = np.mean(embeddings, axis=0)

        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    @staticmethod
    def compute_perspective_direction(
        perspective_texts,
        embeddings_dict,
        reference_texts=None,
        centroid_method="mean",
        trim_proportion=0.1,
        huber_delta=1.0,
        ref_weight=1.0,
    ):
        """
        Compute a direction representing a given perspective.

        - If `reference_texts` is provided and non-empty: return the normalized vector
          from the reference centroid to the perspective centroid.
        - If no `reference_texts`: return the normalized centroid of the perspective texts
          (suitable for cosine-similarity style projection).
        """
        if not perspective_texts:
            return None

        norm_embeddings = DirectionCalculator.normalize_embeddings(embeddings_dict)

        perspective_centroid = DirectionCalculator.compute_group_centroid(
            perspective_texts,
            norm_embeddings,
            method=centroid_method,
            trim_proportion=trim_proportion,
            huber_delta=huber_delta,
        )
        if perspective_centroid is None:
            return None

        if reference_texts:
            reference_centroid = DirectionCalculator.compute_group_centroid(
                reference_texts,
                norm_embeddings,
                method=centroid_method,
                trim_proportion=trim_proportion,
                huber_delta=huber_delta,
            )
            if reference_centroid is None:
                return None
            direction = perspective_centroid - float(ref_weight) * reference_centroid
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 0 else direction

        # No reference: treat the (normalized) perspective centroid itself as the direction
        return perspective_centroid
