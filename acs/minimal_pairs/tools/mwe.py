import json

with open(
    "data/resources/mwes.json",
    "r",
    encoding="utf-8",
) as f:
    english_mwes = json.load(f)

mwe_bank = set()

for word in english_mwes:
    for exp in english_mwes[word]:
        mwe_bank.add(exp.lower())


def multiword_search(tokens):

    n = len(tokens)
    tags = ["O"] * n

    i = 0
    while i < n:
        match_found = False
        for j in range(n, i, -1):
            phrase = " ".join(tokens[i:j]).lower()
            if phrase in mwe_bank:
                for k in range(i, j):
                    tags[k] = "I"
                i = i + 1
                match_found = True
                break
        if not match_found:
            i += 1

    return tags
