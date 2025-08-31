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
    def compute_group_centroid(texts, embeddings_dict):
        """
        Compute and return the centroid of a list of texts using embeddings.
        """
        if not texts:
            return None
        embeddings = [embeddings_dict[text] for text in texts]
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    @staticmethod
    def compute_perspective_direction(perspective_texts, embeddings_dict, reference_texts=None):
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

        perspective_centroid = DirectionCalculator.compute_group_centroid(perspective_texts, norm_embeddings)
        if perspective_centroid is None:
            return None

        if reference_texts:
            reference_centroid = DirectionCalculator.compute_group_centroid(reference_texts, norm_embeddings)
            if reference_centroid is None:
                return None
            direction = perspective_centroid - reference_centroid
            norm = np.linalg.norm(direction)
            return direction / norm if norm > 0 else direction

        # No reference: treat the (normalized) perspective centroid itself as the direction
        return perspective_centroid
