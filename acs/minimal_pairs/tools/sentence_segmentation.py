import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel,
                          AutoModelForTokenClassification, AutoTokenizer,
                          XLMRobertaConfig)
from transformers.modeling_outputs import \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.canine.modeling_canine import TokenClassifierOutput
from transformers.models.xlm_roberta import (XLMRobertaForTokenClassification,
                                             XLMRobertaModel)
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaEmbeddings, XLMRobertaEncoder, XLMRobertaPooler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubwordXLMConfig(XLMRobertaConfig):
    """Config for XLM-R and XLM-V models. Used for token-level training.

    Args:
        XLMRobertaConfig: Base class.
    """

    model_type = "xlm-token"
    mixture_name = "xlm-token"

    def __init__(
        self,
        lookahead=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mixture_name = "xlm-token"
        self.lookahead = lookahead


AutoConfig.register("xlm-token", SubwordXLMConfig)


class SubwordXLMForTokenClassification(XLMRobertaForTokenClassification):
    config_class = SubwordXLMConfig

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, threshold=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SubwordXLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.threshold = threshold

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hashed_ids: Optional[torch.Tensor] = None,
        language_ids=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.threshold is not None:
            logits = torch.sigmoid(logits)
            if logits.shape[2] > 1:
                logits = logits[:, :, :1]
            logits = (logits > self.threshold).float()
            logits = torch.cat((1 - logits, logits), dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SubwordXLMRobertaModel(XLMRobertaModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->XLMRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)

        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None
        self.effective_lookahead = (
            config.lookahead // config.num_hidden_layers
            if config.lookahead is not None
            else None
        )

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.effective_lookahead
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int],
        lookahead: Optional[int] = None,
        device: torch.device = None,
        dtype: torch.float = None,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.",
                    FutureWarning,
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = (
                    ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                        input_shape, attention_mask, device
                    )
                )
            if lookahead:
                # lookahead mask of shape [batch_size, 1, seq_length, seq_length]
                # the current token should attend to the next `lookahead` tokens
                # the current token should not attend to the previous `lookahead` tokens
                _, seq_length = attention_mask.shape
                # Create a lookahead mask
                lookahead_mask = torch.tril(
                    torch.ones(seq_length, seq_length), diagonal=lookahead, out=None
                ).to(attention_mask.device)
                # Combine the attention mask with the lookahead mask
                extended_attention_mask = (
                    attention_mask[:, None, None, :] * lookahead_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            dtype
        ).min
        return extended_attention_mask


AutoModel.register(SubwordXLMConfig, SubwordXLMForTokenClassification)
AutoModelForTokenClassification.register(
    SubwordXLMConfig, SubwordXLMForTokenClassification
)

if __name__ == "__main__":
    # test XLM
    from transformers import AutoConfig, AutoTokenizer

    model_str = "xlm-roberta-base"
    config = AutoConfig.from_pretrained(model_str)
    config.num_labels = 4
    config.num_hidden_layers = 1
    backbone = SubwordXLMForTokenClassification.from_pretrained(
        model_str, config=config
    )
    print(summary(backbone, depth=4))

    # some sample input
    text = "A sentence. Now we move on. And on and this is the last sentence. Now, we are starting to move on to the next sentence. This is the last sentence."
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        pad_to_multiple_of=512,
        padding=True,
    )
    from tokenizers import AddedToken

    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    print(tokenizer.tokenize(text))
    print(tokenizer.encode(text))
    print(tokens)

    # forward pass
    print(backbone(**tokens))


class SegmentationDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokenized_output = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        tokenized_text = tokenized_output.input_ids.squeeze(0)  
        offset_mapping = tokenized_output.offset_mapping.squeeze(
            0
        ) 
        return tokenized_text, offset_mapping


@torch.no_grad()
def get_sentence_segmentation_token_labels(texts, batch_size=128):

    segmentation_model_checkpoint = "segment-any-text/sat-12l-sm"

    segmentation_model = SubwordXLMForTokenClassification.from_pretrained(
        segmentation_model_checkpoint, threshold=0.3
    ).to(device)

    id2label = {0: "O", 1: "|"}
    label2id = {v: k for k, v in id2label.items()}

    segmentation_model.config.id2label = id2label
    segmentation_model.config.label2id = label2id

    segmentation_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    dataset = SegmentationDataset(texts, segmentation_tokenizer)

    def custom_collate_fn(batch):
        input_ids, offset_mappings = zip(*batch)
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=segmentation_tokenizer.pad_token_id,
        )
        return input_ids_padded, offset_mappings

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    segmented_texts = []

    num_done = 0

    for batch_index, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids, offset_mappings = batch

        input_ids = input_ids.to(segmentation_model.device)
        attention_mask = (input_ids != segmentation_tokenizer.pad_token_id).to(
            segmentation_model.device
        )

        logits = segmentation_model(input_ids, attention_mask=attention_mask).logits

        predictions = torch.argmax(logits, dim=2)

        for i, text in enumerate(
            texts[batch_index * batch_size : (batch_index + 1) * batch_size]
        ):

            segmented_tweet = []

            predicted_subword_labels = [
                segmentation_model.config.id2label[t.item()] for t in predictions[i]
            ]

            offset_mapping = offset_mappings[i].tolist()

            start = 0

            for predicted_subword_label, offsets in zip(
                predicted_subword_labels, offset_mapping
            ):
                if offsets[0] == offsets[1]:
                    continue

                if predicted_subword_label == "|":
                    segmented_tweet.append(text[start : offsets[1]].strip())
                    start = offsets[1]

            if start < len(text):
                segmented_tweet.append(text[start:].strip())

            segmented_texts.append(segmented_tweet)

        num_done += len(batch)

    return segmented_texts


def get_offsets(text, tokens):
    """Find the character offsets of each token in the original text."""
    offsets = []
    start = 0
    for token in tokens:
        start = text.find(token, start)
        if start == -1:  # In case a token cannot be found (should ideally not happen)
            if token.startswith("#"):
                try_tokens = "# " + token[1:]
                start = text.find(try_tokens, start)
            else:
                return None
        end = start + len(token)
        offsets.append((start, end))
        start = end  # Move start to end for next search
    return offsets


def segment_labels(text, tokens, labels, segmented_text):
    """Segment the tokens and labels according to the segmented_paragraph."""
    offsets = get_offsets(text, tokens)

    if offsets is None:
        return None, None

    segmented_tokens = []
    segmented_labels = []

    for sentence in segmented_text:
        sentence_tokens = []
        sentence_labels = []
        sentence_start = text.find(sentence)
        sentence_end = sentence_start + len(sentence)

        for offset, token, label in zip(offsets, tokens, labels):
            if offset[0] >= sentence_start and offset[1] <= sentence_end:
                # Token is within the current sentence
                sentence_tokens.append(token)
                sentence_labels.append(label)

        segmented_tokens.append(sentence_tokens)
        segmented_labels.append(sentence_labels)

    return segmented_tokens, segmented_labels
