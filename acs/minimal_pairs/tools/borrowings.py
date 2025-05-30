import json

import emoji


def add_extra_labels(words, token_labels, lang1, lang2="en"):

    borrowings_path = "data/resources/borrowings.json"

    with open(borrowings_path, "r", encoding="utf-8") as file:
        borrowings = json.load(file)

    assert len(words) == len(token_labels)

    for i in range(len(token_labels)):
        if bool(emoji.emoji_count(words[i])):
            token_labels[i] = "o"

    for i in range(len(token_labels)):
        if (
            words[i] in borrowings[lang1][lang2]
            or words[i].lower() in borrowings[lang1][lang2]
        ) and len(words[i]) > 3:

            if token_labels[i] != "lang1":
                continue

            if (
                (i == 0 and len(token_labels) > 1 and token_labels[i + 1] == "lang2")
                or (
                    i == len(token_labels) - 1
                    and len(token_labels) > 1
                    and token_labels[i - 1] == "lang2"
                )
                or (
                    i > 0
                    and i < len(token_labels) - 1
                    and (
                        token_labels[i - 1] == "lang2" or token_labels[i + 1] == "lang2"
                    )
                )
            ):
                token_labels[i] = "lang2.b"
            else:
                token_labels[i] = "b.lang2"

    for i in range(len(token_labels)):
        if (
            words[i] in borrowings[lang2][lang1]
            or words[i].lower() in borrowings[lang2][lang1]
        ) and len(words[i]) > 3:

            if token_labels[i] != "lang2":
                continue

            if (
                (i == 0 and token_labels[i + 1] == "lang1")
                or (i == len(token_labels) - 1 and token_labels[i - 1] == "lang1")
                or (
                    i > 0
                    and i < len(token_labels) - 1
                    and (
                        token_labels[i - 1] == "lang1" or token_labels[i + 1] == "lang1"
                    )
                )
            ):
                token_labels[i] = "lang1.b"
            else:
                token_labels[i] = "b.lang1"

    island = []

    for i in range(len(token_labels)):

        word = words[i]
        label = token_labels[i]

        # Capitalized words

        if len(island) > 1 and label != island[-1][1]:
            island_indices = [i[0] for i in island]
            island_labels = [i[1] for i in island]
            island_words = [i[2] for i in island]
            assert len(set(island_labels)) == 1
            num_capitalized = sum([1 for word in island_words if word[0].isupper()])
            proportion_capitalized = num_capitalized / len(island_words)
            if proportion_capitalized >= 0.75:
                for index in island_indices:
                    token_labels[index] = "ne." + token_labels[index]
            island = []

        if label not in ["lang1", "lang2"]:
            island = []
        elif len(island) == 1 and label != island[-1][1]:
            island = [(i, label, word)]
        else:
            island.append((i, label, word))

    if island:
        island_indices = [i[0] for i in island]
        island_labels = [i[1] for i in island]
        island_words = [i[2] for i in island]
        assert len(set(island_labels)) == 1
        num_capitalized = sum([1 for word in island_words if word[0].isupper()])
        proportion_capitalized = num_capitalized / len(island_words)
        if proportion_capitalized >= 0.75:
            for index in island_indices:
                token_labels[index] = "ne." + token_labels[index]

    return token_labels
