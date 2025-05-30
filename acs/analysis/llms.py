import argparse
import json
import os

from datasets import load_dataset
from minicons import scorer
from tqdm import tqdm

RESULTS_DIR = "data/results/"


def create_batches(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def process_batches(sentences, model, llm_type, batch_size):
    batches = create_batches(sentences, batch_size)
    log_probs = []

    for batch in batches:
        if llm_type == "masked":
            batch_scores = model.sequence_score(
                batch,
                reduction=lambda x: -x.sum(0).item(),
                PLL_metric="within_word_l2r",
            )
        elif llm_type == "incremental":
            batch_scores = model.sequence_score(
                batch,
                reduction=lambda x: -x.sum(0).item(),
            )
        log_probs.extend(batch_scores)
    return log_probs


def evaluate_llm(model, llm_type, minimal_pairs, batch_size=10):
    observed_sentences = [mp["observed_sentence"] for mp in minimal_pairs]
    manipulated_sentences = [mp["manipulated_sentence"] for mp in minimal_pairs]

    observed_log_probs = process_batches(
        observed_sentences, model, llm_type, batch_size
    )
    manipulated_log_probs = process_batches(
        manipulated_sentences, model, llm_type, batch_size
    )

    return observed_log_probs, manipulated_log_probs


def run_evaluation(models, llm_type, scorer_class, lang):
    for model_id in tqdm(models, desc=f"{llm_type} models"):
        model = scorer_class(model_id, "cuda")

        os.makedirs(os.path.join(RESULTS_DIR, lang), exist_ok=True)
        save_name = model_id.split("/")[-1]
        RESULT_PATH = os.path.join(RESULTS_DIR, lang, save_name + ".jsonl")

        ds = load_dataset("igorsterner/acs-benchmark", data_dir=lang, split="test")
        minimal_pairs = list(ds)

        observed_log_probs, manipulated_log_probs = evaluate_llm(
            model, llm_type, minimal_pairs
        )

        with open(RESULT_PATH, "w") as f:
            for mp, real, manipulated in zip(
                minimal_pairs, observed_log_probs, manipulated_log_probs
            ):
                row = {
                    "id": mp["id"],
                    "observed_logprob": real,
                    "manipulated_logprob": manipulated,
                }
                f.write(json.dumps(row) + "\n")

        scores = [
            1 if real < manipulated else 0
            for real, manipulated in zip(observed_log_probs, manipulated_log_probs)
        ]
        accuracy = sum(scores) / len(scores)
        print(f"{save_name} ({lang}): Accuracy = {accuracy:.2%}%")

    if "model" in locals():
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", required=True)
    parser.add_argument("--masked_lms", nargs="+", default=[])
    parser.add_argument("--incremental_lms", nargs="+", default=[])
    args = parser.parse_args()

    for lang in args.langs:
        if args.incremental_lms:
            run_evaluation(
                args.incremental_lms,
                "incremental",
                scorer.IncrementalLMScorer,
                lang,
            )
        if args.masked_lms:
            run_evaluation(
                args.masked_lms,
                "masked",
                scorer.MaskedLMScorer,
                lang,
            )


if __name__ == "__main__":
    main()
