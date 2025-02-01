import itertools

import matplotlib.pyplot as plt
import numpy as np

from clustering import DirectionalClusterer
from direction_calculator import DirectionCalculator
from embedding_processor import EmbeddingProcessor
from text_pairer import TextPairer


def main():
    # 1. Input texts
    texts = [
        "I love dogs and playing outside.",
        "Dogs are great companions.",
        "I enjoy spending time with my pet dog.",
        "Cats make cozy companions.",
        "I adore cats.",
        "There is heavy rain today.",
        "The weather is gloomy with rain.",
        "I love a sunny day at the park.",
    ]

    # 2. Get embeddings
    embedder = EmbeddingProcessor()
    embeddings = embedder.encode(texts)
    embeddings_dict = {text: emb for text, emb in zip(texts, embeddings)}

    # 3. Identify groups
    pairer = TextPairer(texts)
    dog_texts = [text for text in texts if pairer.assign_group(text) == "dogs"]
    cat_texts = [text for text in texts if pairer.assign_group(text) == "cats"]
    other_texts = [text for text in texts if pairer.assign_group(text) == "others"]

    # 4. Compute directional embeddings
    # For dog direction: dogs are positive, cats+others are negative
    dog_direction = DirectionCalculator.compute_semantic_direction(
        positive_texts=dog_texts, negative_texts=cat_texts + other_texts, embeddings_dict=embeddings_dict
    )

    # For cat direction: cats are positive, dogs+others are negative
    cat_direction = DirectionCalculator.compute_semantic_direction(
        positive_texts=cat_texts, negative_texts=dog_texts + other_texts, embeddings_dict=embeddings_dict
    )

    # Store directions
    directions = {"dog": dog_direction, "cat": cat_direction}

    # After getting embeddings, add dictionary learning
    clusterer = DirectionalClusterer(positive_threshold=0.2, negative_threshold=-0.2)
    transformed, components = clusterer.learn_dictionary(embeddings, n_components=2)

    # Use the learned components as additional directions
    directions["learned_1"] = components[0]
    directions["learned_2"] = components[1]

    # 5. Project all texts onto each direction
    projections = {}
    analyses = {}
    for direction_name, direction in directions.items():
        if direction is not None:
            proj = np.dot(embeddings, direction)
            projections[direction_name] = {text: p for text, p in zip(texts, proj)}

            # Analyze projections
            proj_values = np.array(list(projections[direction_name].values()))
            group_labels = [pairer.assign_group(text) for text in texts]
            analyses[direction_name] = clusterer.analyze_projections(texts, proj_values, group_labels)

    # 6. Visualization
    plt.figure(figsize=(15, 10))

    # Helper function to create sorted projection plots
    def plot_projection(ax, direction_name, title):
        if direction_name in projections:
            proj_dict = projections[direction_name]
            # Sort texts by their group
            dog_projs = [(i, p) for i, (t, p) in enumerate(proj_dict.items()) if pairer.assign_group(t) == "dogs"]
            cat_projs = [(i, p) for i, (t, p) in enumerate(proj_dict.items()) if pairer.assign_group(t) == "cats"]
            other_projs = [(i, p) for i, (t, p) in enumerate(proj_dict.items()) if pairer.assign_group(t) == "others"]

            # Plot each group with different colors
            if dog_projs:
                ax.scatter([x[0] for x in dog_projs], [x[1] for x in dog_projs], c="red", label="Dogs", s=100)
            if cat_projs:
                ax.scatter([x[0] for x in cat_projs], [x[1] for x in cat_projs], c="blue", label="Cats", s=100)
            if other_projs:
                ax.scatter([x[0] for x in other_projs], [x[1] for x in other_projs], c="gray", label="Others", s=100)

            ax.set_title(title)
            ax.set_xlabel("Text Index")
            ax.set_ylabel("Projection Value")
            ax.legend()

    # Create subplots for each direction
    plt.subplot(2, 2, 1)
    plot_projection(plt.gca(), "dog", "Dog Direction")

    plt.subplot(2, 2, 2)
    plot_projection(plt.gca(), "cat", "Cat Direction")

    plt.subplot(2, 2, 3)
    plot_projection(plt.gca(), "learned_1", "Learned Direction 1")

    plt.subplot(2, 2, 4)
    plot_projection(plt.gca(), "learned_2", "Learned Direction 2")

    plt.tight_layout()
    plt.show()

    # Print numerical results
    for direction_name in ["dog", "cat", "learned_1", "learned_2"]:
        print(f"\nProjections on {direction_name.capitalize()} Direction:")
        if direction_name in projections:
            sorted_items = sorted(projections[direction_name].items(), key=lambda x: x[1], reverse=True)
            for text, proj in sorted_items:
                print(f"{text[:30]:30} : {proj:.4f} ({pairer.assign_group(text)})")

            # Print analysis
            print(f"\nAnalysis for {direction_name.capitalize()} Direction:")
            analysis = analyses[direction_name]
            print("Overall stats:", analysis["projection_stats"])
            print("\nGroup stats:")
            for group, stats in analysis["group_stats"].items():
                print(f"{group}: {stats}")


if __name__ == "__main__":
    main()
