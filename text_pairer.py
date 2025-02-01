class TextPairer:
    def __init__(self, texts):
        self.texts = texts
        # Cache of group assignments to avoid repeated computation.
        self.group_assignments = {}
        self.grouped_texts = {}

    def assign_group(self, text):
        """
        Assign a text to group 'dogs', 'cats', or 'others'
        based on whether it mentions 'dog' or 'cat'.
        """
        if text in self.group_assignments:
            return self.group_assignments[text]

        text_lower = text.lower()
        # If text mentions dog but not cat, then group as "dogs"
        if "dog" in text_lower and "cat" not in text_lower:
            group = "dogs"
        # If text mentions cat but not dog, then group as "cats"
        elif "cat" in text_lower and "dog" not in text_lower:
            group = "cats"
        # Otherwise, assign to "others" (no relevant mention or ambiguous)
        else:
            group = "others"

        self.group_assignments[text] = group
        return group

    def group_texts(self):
        """
        Group texts into dictionary by their group.
        """
        # Clear previous groups if any.
        self.grouped_texts = {}
        for text in self.texts:
            group = self.assign_group(text)
            if group not in self.grouped_texts:
                self.grouped_texts[group] = []
            self.grouped_texts[group].append(text)
        return self.grouped_texts

    def get_similar_pairs(self):
        """
        Generate similar pairs: texts within the same group.
        """
        similar_pairs = []
        groups = self.group_texts()
        for group, group_texts in groups.items():
            if len(group_texts) > 1:
                # Generate all unique pairs within the same group.
                import itertools

                similar_pairs.extend(list(itertools.combinations(group_texts, 2)))
        return similar_pairs

    def get_dissimilar_pairs(self):
        """
        Generate dissimilar pairs: texts from different groups.
        """
        dissimilar_pairs = []
        groups = self.group_texts()
        group_names = list(groups.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                for t1 in groups[group_names[i]]:
                    for t2 in groups[group_names[j]]:
                        dissimilar_pairs.append((t1, t2))
        return dissimilar_pairs
