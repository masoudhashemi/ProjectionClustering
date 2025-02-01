import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import DictionaryLearning


class Clustering:
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def kmeans_cluster(self, data):
        """
        Cluster the data using KMeans.
        Accepts data as a 1D list (reshaped) or multi-dimensional array.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return kmeans.fit_predict(np.array(data))

    def dictionary_learning(self, embeddings, n_components=2):
        """
        Learn a dictionary representation from embeddings.
        """
        dict_learner = DictionaryLearning(n_components=n_components, random_state=self.random_state)
        return dict_learner.fit_transform(embeddings)


class DirectionalClusterer:
    def __init__(self, positive_threshold=0.2, negative_threshold=-0.2):
        """
        Initialize with thresholds for classifying projection values.
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.dict_learner = None

    def learn_dictionary(self, embeddings, n_components=2):
        """
        Learn dictionary components from embeddings
        """
        dict_learner = DictionaryLearning(n_components=n_components, transform_algorithm="lasso_lars", random_state=42)
        transformed = dict_learner.fit_transform(embeddings)
        self.dict_learner = dict_learner
        return transformed, dict_learner.components_

    def cluster_by_projection(self, projection_values):
        """
        Cluster texts based on their projection values.
        Returns:
          1: Strong positive projection (> positive_threshold)
          0: Neutral projection (between thresholds)
         -1: Strong negative projection (< negative_threshold)
        """
        clusters = []
        for value in projection_values:
            if value > self.positive_threshold:
                clusters.append(1)  # Strong positive
            elif value < self.negative_threshold:
                clusters.append(-1)  # Strong negative
            else:
                clusters.append(0)  # Neutral
        return np.array(clusters)

    def analyze_projections(self, texts, projections, group_labels):
        """
        Analyze the separation quality based on the projected values.
        Returns:
         - clusters (the threshold-based classification)
         - overall statistics for the projection values
         - per-group statistics
        """
        clusters = self.cluster_by_projection(projections)

        analysis = {
            "clusters": clusters,
            "projection_stats": {
                "mean": np.mean(projections),
                "std": np.std(projections),
                "min": np.min(projections),
                "max": np.max(projections),
            },
            "group_stats": {},
        }

        unique_groups = set(group_labels)
        for group in unique_groups:
            # Get projections for texts in this group.
            group_indices = [i for i, g in enumerate(group_labels) if g == group]
            group_projections = projections[group_indices]
            analysis["group_stats"][group] = {
                "mean": np.mean(group_projections),
                "std": np.std(group_projections),
                "min": np.min(group_projections),
                "max": np.max(group_projections),
                "count": len(group_projections),
            }

        return analysis
