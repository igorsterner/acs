import gzip
import json
from collections import defaultdict
from io import BytesIO, TextIOWrapper
from pathlib import Path

import requests
from tqdm import tqdm

all_borrowings = defaultdict(lambda: defaultdict(set))


url = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"

response = requests.get(url, stream=True)
response.raise_for_status()
data_bytes = BytesIO(response.content)

with gzip.open(data_bytes, mode="rt", encoding="utf-8") as f:
    for i, line in tqdm(enumerate(f), desc="Extracting..."):
        data = json.loads(line)

        if "word" in data and "etymology_templates" in data:
            for template in data["etymology_templates"]:
                if template["name"] == "bor" or template["name"] == "ubor":
                    lang1 = template["args"]["1"]
                    lang2 = template["args"]["2"]
                    if lang1 != lang2:
                        all_borrowings[lang1][lang2].add(data["word"])

for lang1 in all_borrowings:
    for lang2 in all_borrowings[lang1]:
        all_borrowings[lang1][lang2] = list(all_borrowings[lang1][lang2])

with open(
    "data/resources/borrowings.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(all_borrowings, f, indent=4, ensure_ascii=False)
