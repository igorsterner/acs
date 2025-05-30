import stanza
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_lang1(data, lang1):

    lang_code_map = {"in": "id"}
    lang1 = lang_code_map.get(lang1, lang1)

    nlp = stanza.Pipeline(
        lang=lang1,
        processors="tokenize",
        tokenize_no_ssplit=True,
        use_gpu=True,
        device=device,
    )

    text_ids = list(data.keys())

    batch_size = 1000

    for i in tqdm(range(0, len(text_ids), batch_size), desc="Batch tokenizing"):

        batch_ids = text_ids[i : i + batch_size]

        texts = [data[id]["text"] for id in batch_ids]

        stanza_docs = nlp.bulk_process(texts)

        for text_id, doc in zip(batch_ids, stanza_docs):
            tokens = []
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    tokens.append(token.text)

            data[text_id]["tokens"] = tokens

    return data
