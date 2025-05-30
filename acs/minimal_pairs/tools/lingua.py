from lingua import LanguageDetectorBuilder


def lingua_filtering(data, lang1):

    ids = list(data.keys())

    texts_lang1 = [
        " ".join(
            token
            for token, label in zip(data[id]["tokens"], data[id]["langs"])
            if label == "lang1"
        )
        for id in ids
    ]

    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()

    lingua_langs = detector.detect_languages_in_parallel_of(texts_lang1)

    for i, (id, lang) in enumerate(zip(ids, lingua_langs)):
        if lang is None:
            del data[id]
        else:
            if lang.iso_code_639_1.name.lower() != lang1:
                del data[id]

    return data
