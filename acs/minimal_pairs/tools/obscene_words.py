# this is the script used to remove obscene words. You will need to download the data from
# https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

import os

from tqdm import tqdm


def multiword_search(tokens, bank):
    n = len(tokens)

    i = 0
    while i < n:
        match_found = False
        for j in range(n, i, -1):
            phrase = " ".join(tokens[i:j]).lower()
            if phrase in bank:
                return True
        if not match_found:
            i += 1

    return False


def remove_obscene(data, lang1, lang2):

    lang1_file = (
        "data/resources/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/" + lang1
    )
    lang2_file = (
        "data/resources/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/" + lang2
    )

    # check if file exists
    if os.path.isfile(lang1_file):
        with open(lang1_file, "r", encoding="utf-8") as file:
            lang1_phrases = file.read().splitlines()
    else:
        print(f"File {lang1_file} not found.")
        lang1_phrases = []

    with open(lang2_file, "r", encoding="utf-8") as file:
        lang2_phrases = file.read().splitlines()

    for id in tqdm(list(data.keys()), desc="Removing obscene..."):
        if multiword_search(data[id]["tokens"], lang1_phrases):
            del data[id]
        elif multiword_search(data[id]["tokens"], lang2_phrases):
            del data[id]

    return data
