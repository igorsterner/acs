import argparse
import json
import os
import pickle
import types

import editdistance
import stanza
import torch
from acs.minimal_pairs.tools import (alignment, borrowings,
                                                   language_identification,
                                                   lingua, mwe,
                                                   other_processing,
                                                   sentence_segmentation,
                                                   tokenization, translation)
from stanza.models.classifiers.utils import ExtraVectors, ModelType, WVType

# this is a workaround for issue with torch serialization and stanza
torch.serialization.add_safe_globals([types.SimpleNamespace])
torch.serialization.add_safe_globals([ExtraVectors])
torch.serialization.add_safe_globals([WVType])
torch.serialization.add_safe_globals([ModelType])


def main():

    #  == LOAD DATA (without duplicates) ==

    INPUT_DATA_PATH = f"data/demonstration_example/{args.lang1}-{args.lang2}.jsonl"

    all_data = {}
    seen = set()
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data["text"] not in seen:
                all_data[data["id"]] = data
                seen.add(data["text"])

    #  == tokenization ==

    all_data = tokenization.tokenize_lang1(all_data, args.lang1)

    #  == MWEs ==

    for id in list(all_data.keys()):
        all_data[id]["mwe_bio"] = mwe.multiword_search(all_data[id]["tokens"])

    #  == token-level language identification ==

    all_data = language_identification.token_identify(all_data, "igorsterner/AnE-LID")
    all_data = language_identification.token_identify(all_data, "igorsterner/AnE-NER")

    # == little things ==

    for id in list(all_data.keys()):
        all_data[id]["tokens"] = other_processing.merge_mentions(all_data[id]["tokens"])
        all_data[id]["text"] = all_data[id]["text"].replace(" ", " ")
        all_data[id]["langs"] = borrowings.add_extra_labels(
            all_data[id]["tokens"], all_data[id]["langs"], args.lang1, args.lang2
        )
        all_data[id]["langs"] = [
            "o" if (token in ["@USER", "HTTPURL"]) else label
            for token, label in zip(all_data[id]["tokens"], all_data[id]["langs"])
        ]
        all_data[id]["langs"] = [
            "ht." + label if token.startswith("#") else label
            for token, label in zip(all_data[id]["tokens"], all_data[id]["langs"])
        ]

    # == Lingua step ==

    all_data = lingua.lingua_filtering(all_data, args.lang1)

    # new data format from now onwards
    samples_data = [{"id": id, **all_data[id]} for id in all_data]

    #  == segment == --

    texts = [data["text"] for data in samples_data]

    segmented_texts = sentence_segmentation.get_sentence_segmentation_token_labels(
        texts, batch_size=128
    )

    segmented_data = []
    seen_texts = set()

    skipping_counter = 0

    for i, data in enumerate(samples_data):

        if data["id"].startswith("SAGT"):  # no need to segment tr-de
            segmented_data.append(data)
            continue

        segmented_text = segmented_texts[i]

        segmented_tokens, segmented_labels = sentence_segmentation.segment_labels(
            data["text"], data["tokens"], data["langs"], segmented_text
        )

        if segmented_tokens is None:
            skipping_counter += 1
            continue

        sent_num = 0

        for sentence, tokens, langs in zip(
            segmented_text, segmented_tokens, segmented_labels
        ):
            if sentence in seen_texts:
                continue
            elif not sentence:
                continue

            seen_texts.add(sentence)

            save_data = {
                "id": data["id"] + "_" + str(sent_num),
                "text": sentence,
                "tokens": tokens,
                "langs": langs,
                "mwe_bio": data["mwe_bio"],
            }

            sent_num += 1

            if "original_text" in data:
                save_data["original_text"] = data["original_text"]
            else:
                save_data["original_text"] = data["text"]

            segmented_data.append(save_data)

    #  == translate == -

    texts = [data["text"] for data in segmented_data]

    texts_lang1 = translation.madlad_translate(
        texts, "jbochi/madlad400-3b-mt", args.lang1, None, batch_size=128
    )

    texts_lang2 = translation.madlad_translate(
        texts_lang1, "jbochi/madlad400-3b-mt", args.lang2, None, batch_size=128
    )

    assert len(texts) == len(texts_lang1) == len(texts_lang2)

    for i, data in enumerate(segmented_data):
        data["text_lang1"] = texts_lang1[i]
        data["text_lang2"] = texts_lang2[i]

    segmented_data = [
        data
        for data in segmented_data
        if len(data["text"]) < 200 and len(data["tokens"]) > 5
    ]
    segmented_data = [
        data
        for data in segmented_data
        if editdistance.eval(data["text"].lower(), data["text_lang2"].lower()) > 5
    ]
    segmented_data = [
        data
        for data in segmented_data
        if editdistance.eval(data["text"].lower(), data["text_lang1"].lower()) > 5
    ]

    #  == parse into universal dependencies ==

    stanza_map = {
        "zh": "zh-hans",
    }

    nlp_lang1 = stanza.Pipeline(
        stanza_map.get(args.lang1, args.lang1),
        tokenize_no_ssplit=True,
    )

    out_docs = nlp_lang1.bulk_process(texts_lang1)

    for i, data in enumerate(segmented_data):
        if len(out_docs[i].sentences) == 0:
            data["doc_lang1"] = None
        else:
            assert len(out_docs[i].sentences) == 1, out_docs[i]
            data["doc_lang1"] = out_docs[i].sentences[0]

    nlp_lang2 = stanza.Pipeline(
        args.lang2,
        tokenize_no_ssplit=True,
    )

    out_docs = nlp_lang2.bulk_process(texts_lang2)
    for i, data in enumerate(segmented_data):
        if len(out_docs[i].sentences) == 0:
            data["doc_lang2"] = None
        else:
            assert len(out_docs[i].sentences) == 1, out_docs[i]
            data["doc_lang2"] = out_docs[i].sentences[0]

    for data in segmented_data:
        data["words_lang2"] = [word.text for word in data["doc_lang2"].words]
        data["upos_lang2"] = [word.upos for word in data["doc_lang2"].words]
        data["xpos_lang2"] = [word.xpos for word in data["doc_lang2"].words]
        data["dependencies_lang2"] = [
            (head.id - 1, deprel, dep.id - 1)
            for head, deprel, dep in data["doc_lang2"].dependencies
            if head.id != 0 and dep.id != 0
        ]
        data["ner_lang2"] = other_processing.get_ner_bio_labels(data["doc_lang2"])

        data["words_lang1"] = [word.text for word in data["doc_lang1"].words]
        data["upos_lang1"] = [word.upos for word in data["doc_lang1"].words]
        data["xpos_lang1"] = [word.xpos for word in data["doc_lang1"].words]
        data["dependencies_lang1"] = [
            (head.id - 1, deprel, dep.id - 1)
            for head, deprel, dep in data["doc_lang1"].dependencies
            if head.id != 0 and dep.id != 0
        ]
        data["ner_lang1"] = other_processing.get_ner_bio_labels(data["doc_lang1"])

    segmented_data = [
        data
        for data in segmented_data
        if "X" not in data["upos_lang1"] and "X" not in data["upos_lang2"]
    ]

    #  == align in both directions ==

    for data in segmented_data:
        words_lang1 = [word.text for word in data["doc_lang1"].words]
        data["words_lang1"] = words_lang1

        words_lang2 = [word.text for word in data["doc_lang2"].words]
        data["words_lang2"] = words_lang2

    input_cs_lang2 = [(data["tokens"], data["words_lang2"]) for data in segmented_data]

    alignments_cs_lang2 = alignment.batch_align(
        input_cs_lang2,
        batch_size=128,
    )

    for data, alignments in zip(segmented_data, alignments_cs_lang2):
        data["alignments_cs_lang2"] = alignments

    input_cs_lang1 = [(data["tokens"], data["words_lang1"]) for data in segmented_data]
    alignments_cs_lang1 = alignment.batch_align(
        input_cs_lang1,
        batch_size=128,
    )

    for data, alignments in zip(segmented_data, alignments_cs_lang1):
        data["alignments_cs_lang1"] = alignments

    for data in segmented_data:
        (
            data["alignments_cs_lang2"],
            data["alignments_cs_lang1"],
            data["alignments_lang2_cs"],
            data["alignments_lang1_cs"],
        ) = other_processing.compute_alignments(
            data["alignments_cs_lang2"],
            data["alignments_cs_lang1"],
            data["tokens"],
            data["words_lang2"],
            data["words_lang1"],
        )

    #  == save ==

    os.makedirs("data/cache/", exist_ok=True)

    with open(
        f"data/cache/processed_data.pkl",
        "wb",
    ) as f:
        pickle.dump(segmented_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang1", type=str, required=True)
    parser.add_argument("--lang2", type=str, required=True)
    args = parser.parse_args()

    assert args.lang1 == args.lang1.lower(), "lang1 should be lower case"
    assert args.lang2 == args.lang2.lower(), "lang2 should be lower case"
    assert args.lang1 != "en", "lang1 should not be english"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
