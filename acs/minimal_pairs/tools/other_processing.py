def merge_mentions(tokens):

    n = len(tokens)
    merged_tokens = []
    i = 0

    while i < n:
        if tokens[i] == "@" and i + 1 < n and tokens[i + 1] == "USER":
            merged_tokens.append("@USER")
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1

    return merged_tokens


def check_adjacent_letters(labels, letter):
    for i in range(len(labels) - 1):
        if labels[i] == letter and labels[i + 1] == letter:
            return True
    return False


def check_adjacent_switch(labels, letter1, letter2):
    for i in range(len(labels) - 1):
        if (labels[i] == letter1 and labels[i + 1] == letter2) or (
            labels[i] == letter2 and labels[i + 1] == letter1
        ):
            return True
    return False


def get_ner_bio_labels(sentence):

    ner_bio = ["O"] * len(sentence.words)

    for token in sentence.tokens:
        ner_label = token.ner
        for word in token.words:
            ner_bio[word.id - 1] = ner_label

    return ner_bio


def compute_alignments(
    alignments_cs_lang2,
    alignments_cs_lang1,
    words_cs,
    words_lang2,
    words_lang1,
):

    alignments_cs_lang1_one_to_many = {i: [] for i in range(len(words_cs))}
    for i, j in alignments_cs_lang1:
        alignments_cs_lang1_one_to_many[i].append(j)

    alignments_lang1_cs = [(v, k) for k, v in alignments_cs_lang1]
    # alignments_lang1_cs = sorted(alignments_lang1_cs, key=lambda x: x[1])
    alignments_lang1_cs_one_to_many = {i: [] for i in range(len(words_lang1))}
    for i, j in alignments_lang1_cs:
        alignments_lang1_cs_one_to_many[i].append(j)

    # alignments_cs_lang2 = sorted(alignments_cs_lang2, key=lambda x: x[1])
    alignments_cs_lang2_one_to_many = {i: [] for i in range(len(words_cs))}
    for i, j in alignments_cs_lang2:
        alignments_cs_lang2_one_to_many[i].append(j)

    alignments_lang2_cs = [(v, k) for k, v in alignments_cs_lang2]
    # alignments_lang2_cs = sorted(alignments_lang2_cs, key=lambda x: x[1])
    alignments_lang2_cs_one_to_many = {i: [] for i in range(len(words_lang2))}
    for i, j in alignments_lang2_cs:
        alignments_lang2_cs_one_to_many[i].append(j)

    return (
        alignments_cs_lang2_one_to_many,
        alignments_cs_lang1_one_to_many,
        alignments_lang2_cs_one_to_many,
        alignments_lang1_cs_one_to_many,
    )
