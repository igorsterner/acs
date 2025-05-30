import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (DataCollatorWithPadding, T5ForConditionalGeneration,
                          T5Tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslationDataset(Dataset):
    def __init__(self, texts, tokenizer, tgt_lang):
        self.texts = [f"<2{tgt_lang}> {text}" for text in texts]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            max_length=150,
        ).input_ids.squeeze(0)
        return {"input_ids": tokenized_text}


@torch.no_grad()
def madlad_translate(texts, model_name, tgt_lang, src_lang=None, batch_size=128):

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    dataset = TranslationDataset(texts, tokenizer, tgt_lang)
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    pred_translations = []
    for batch in tqdm(dataloader, desc=f"Madlad translate to {tgt_lang}"):
        inputs = batch.to(model.device)
        translated_tokens = model.generate(**inputs, max_length=150)
        batch_translations = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        pred_translations.extend(batch_translations)

    return pred_translations
