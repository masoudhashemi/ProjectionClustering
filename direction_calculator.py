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
    def compute_semantic_direction(positive_texts, negative_texts, embeddings_dict):
        """
        Compute a semantic direction that points from the negative examples to the positive examples.
        """
        if not positive_texts or not negative_texts:
            return None

        # Normalize embeddings first.
        norm_embeddings = DirectionCalculator.normalize_embeddings(embeddings_dict)

        # Compute centroids for positive and negative examples.
        positive_centroid = DirectionCalculator.compute_group_centroid(positive_texts, norm_embeddings)
        negative_centroid = DirectionCalculator.compute_group_centroid(negative_texts, norm_embeddings)

        if positive_centroid is None or negative_centroid is None:
            return None

        # Compute the direction from negative centroid to positive centroid.
        direction = positive_centroid - negative_centroid

        # Normalize the computed direction.
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        return direction
