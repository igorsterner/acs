import spacy_alignments as tokenizations
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LangIDDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.text_ids = list(data.keys())
        self.tokens = [data[id]["tokens"] for id in self.text_ids]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        tokenized_output = self.tokenizer(
            self.tokens[idx],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            is_split_into_words=True,
        )
        tokenized_text = tokenized_output.input_ids.squeeze(0)  # Remove batch dimension

        return tokenized_text, self.text_ids[idx]


def get_token_labels(a, b, a_labels):
    a2b, b2a = tokenizations.get_alignments(a, b)

    # Assign labels to subwords
    b_labels = []
    most_common = "o"

    for i, label_indices in enumerate(b2a):

        aligned_subwords = []

        if label_indices:
            for j in label_indices:
                if j < len(a_labels):
                    aligned_subwords.append(a_labels[j])

        if not aligned_subwords:
            aligned_subwords = [most_common]

        most_common = max(set(aligned_subwords), key=aligned_subwords.count)

        b_labels.append(most_common)

    return b_labels


@torch.no_grad()
def token_identify(data, ane_checkpoint, batch_size=128):

    ane_model = AutoModelForTokenClassification.from_pretrained(ane_checkpoint).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(ane_checkpoint)

    dataset = LangIDDataset(data, tokenizer)

    def custom_collate_fn(batch):
        input_ids, text_ids = zip(*batch)
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        return input_ids_padded, text_ids

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    label_mapping = {
        "English": "lang2",  # AnE-LID
        "notEnglish": "lang1",  # AnE-LID
        "Mixed": "m",  # AnE-LID
        "Other": "o",  # AnE-LID
        "I": "ne",  # AnE-NER
        "O": "o",  # AnE-NER
    }

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, text_ids = batch

        input_ids = input_ids.to(ane_model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(ane_model.device)

        logits = ane_model(input_ids, attention_mask=attention_mask).logits

        predictions = torch.argmax(logits, dim=2)

        for i, text_id in enumerate(text_ids):
            prediction = predictions[i].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            subword_labels = [
                ane_model.config.id2label[label_id] for label_id in prediction
            ]
            subword_labels = [label_mapping[label] for label in subword_labels]

            token_labels = get_token_labels(
                tokens, data[text_id]["tokens"], subword_labels
            )

            if "langs" not in data[text_id]:
                # AnE-LID
                data[text_id]["langs"] = token_labels

            else:
                # AnE-NER
                assert len(data[text_id]["langs"]) == len(token_labels)

                data[text_id]["langs"] = [
                    "ne." + label if token_labels[i] == "ne" else label
                    for i, label in enumerate(data[text_id]["langs"])
                ]

    return data
