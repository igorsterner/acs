import json
import string
from pathlib import Path

import requests

URBANDICTIONARY_BASE = "https://raw.githubusercontent.com/mattbierner/urban-dictionary-word-list/master/data"


def urbandictionary2mwe():

    multiwords = []
    entries = []

    for letter in string.ascii_uppercase:
        url = f"{URBANDICTIONARY_BASE}/{letter}.data"
        response = requests.get(url)
        response.raise_for_status()
        entries += response.text.splitlines()

    for entry in entries:
        entry = entry.replace('"', "")

        if len(entry.split()) > 1:
            multiwords.append(entry)

    return multiwords


if __name__ == "__main__":
    en_multiwords = urbandictionary2mwe()

    en_words_in_multiwords = set([w for m in en_multiwords for w in m.split()])

    en_multiword_lookup = {word: set([]) for word in en_words_in_multiwords}

    for m in en_multiwords:
        for w in m.split():
            en_multiword_lookup[w].add(m)

    en_multiword_lookup = {k: list(v) for k, v in en_multiword_lookup.items()}

    with open(
        "data/resources/mwes.json", "w", encoding="utf-8"
    ) as f:
        json.dump(en_multiword_lookup, f)
