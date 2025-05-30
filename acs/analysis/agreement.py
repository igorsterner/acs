import json
from collections import defaultdict

import numpy as np
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


def compute_agreement(lang_pair, participant_id):

    json_path = f"data/human_judgments/{lang_pair}.json"

    assert participant_id in ["0", "1", "2", "3", "4", "all"]

    with open(json_path, "r") as f:
        human_data = json.load(f)

    if participant_id == "all":
        agreement_data = defaultdict(list)
        for participant in human_data:
            for mp_id, is_correct in human_data[participant].items():
                agreement_data[mp_id].append(is_correct)
        agreement_data = list(agreement_data.values())
    else:
        agreement_data = []
        for mp_id, is_correct in human_data[participant_id].items():
            agreement_data.append([is_correct, 1])

    agreement_data = np.array(agreement_data)

    flips = np.random.choice([0, 1], size=len(agreement_data))
    agreement_data = np.where(flips[:, None] == 1, agreement_data, 1 - agreement_data)

    table, _ = aggregate_raters(agreement_data)
    kappa = fleiss_kappa(table, method="fleiss")

    return kappa


if __name__ == "__main__":

    print("de-en")
    kappa = compute_agreement("de-en", "all")
    print(f"  Multirater agreement: {kappa:.2f}")

    for i in range(5):
        kappa = compute_agreement("de-en", str(i))
        print(f"    Singlerater agreement {i}: {kappa:.2f}")

    lang_pairs = [
        "da-en",
        "es-en",
        "fr-en",
        "id-en",
        "it-en",
        "nl-en",
        "sv-en",
        "tr-de",
        "tr-en",
        "zh-en",
    ]
    for pair in lang_pairs:
        print(pair)
        kappa = compute_agreement(pair, "0")
        print(f"    Singlerater agreement: {kappa:.2f}")
