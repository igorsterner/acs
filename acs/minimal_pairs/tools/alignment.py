import itertools

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlignmentDataset(Dataset):
    def __init__(self, data_pairs, tokenizer):
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sent_src, sent_tgt = self.data_pairs[idx]

        token_src = [self.tokenizer.tokenize(word) for word in sent_src]
        token_tgt = [self.tokenizer.tokenize(word) for word in sent_tgt]

        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)),
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )["input_ids"]

        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)),
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )["input_ids"]

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]

        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]

        return (
            ids_src,
            ids_tgt,
            bpe2word_map_src,
            bpe2word_map_tgt,
            len(ids_src),
            len(ids_tgt),
        )


@torch.no_grad()
def batch_align(data_pairs, batch_size=128):
    alignment_model = BertModel.from_pretrained("aneuraz/awesome-align-with-co").to(
        device
    )
    alignment_tokenizer = BertTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

    def collate(examples):
        (
            ids_src,
            ids_tgt,
            bpe2word_map_src,
            bpe2word_map_tgt,
            length_src,
            length_tgt,
        ) = zip(*examples)
        ids_src = torch.nn.utils.rnn.pad_sequence(
            ids_src, batch_first=True, padding_value=alignment_tokenizer.pad_token_id
        )
        ids_tgt = torch.nn.utils.rnn.pad_sequence(
            ids_tgt, batch_first=True, padding_value=alignment_tokenizer.pad_token_id
        )
        return (
            ids_src,
            ids_tgt,
            bpe2word_map_src,
            bpe2word_map_tgt,
            length_src,
            length_tgt,
        )

    dataset = AlignmentDataset(data_pairs, alignment_tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
    )

    align_layer = 8
    threshold = 1e-3

    all_alignments = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, length_src, length_tgt = (
            batch
        )

        ids_src = ids_src.to(device)
        ids_tgt = ids_tgt.to(device)
        attn_mask_src = (ids_src != alignment_tokenizer.pad_token_id).to(device)
        attn_mask_tgt = (ids_tgt != alignment_tokenizer.pad_token_id).to(device)

        out_src = alignment_model(
            ids_src, attention_mask=attn_mask_src, output_hidden_states=True
        )
        out_tgt = alignment_model(
            ids_tgt, attention_mask=attn_mask_tgt, output_hidden_states=True
        )

        batch_alignments = []

        for b in range(len(ids_src)):

            out_src_b = out_src.hidden_states[align_layer][b, : length_src[b]][1:-1]
            out_tgt_b = out_tgt.hidden_states[align_layer][b, : length_tgt[b]][1:-1]

            dot_prod = torch.matmul(out_src_b, out_tgt_b.transpose(-1, -2))

            softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
            softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

            softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

            align_subwords = torch.nonzero(softmax_inter, as_tuple=False)

            alignments = set()
            for i, j in align_subwords:
                src_idx = bpe2word_map_src[b][i]
                tgt_idx = bpe2word_map_tgt[b][j]

                alignments.add((src_idx, tgt_idx))

            batch_alignments.append(alignments)

        all_alignments.extend(batch_alignments)

    return all_alignments
