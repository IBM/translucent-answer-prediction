import torch
import torch.nn as nn
from tap2.bottom_machine.hypers import HypersSF

from pytorch_pretrained_bert.modeling import (
    PreTrainedBertModel,
    BertEncoder,
    BertModel,
    BertConfig
)

import logging

logger = logging.getLogger(__name__)


class PassageEncoder(nn.Module):
    """
    Encodes a single 'chunk' (question + some number of passages).
    Returns the passages' encoding without the question.
    """
    def __init__(self, hypers: HypersSF, pretrained_state_dict=None):
        super(PassageEncoder, self).__init__()
        self.hypers = hypers
        self.bert = BertModel.from_pretrained(
            hypers.bert_model, cache_dir=hypers.bert_archive, no_pooler=True, state_dict=pretrained_state_dict
        )
        self.dropout = nn.Dropout(p=hypers.dropout, inplace=False)
        self.fc_2 = nn.Linear(hypers.bert_hidden_size, hypers.transformer_hidden_size)

    def construct_binary_mask(self, tensor_in, padding_index=0):
        """For bert. 1 denotes actual data and 0 denotes padding"""
        mask = tensor_in != padding_index
        return mask

    def forward(self, sequences, segment_id, qlen):

        sentence_mask = self.construct_binary_mask(sequences)

        paragraph_encoding, _ = self.bert(
            sequences,
            token_type_ids=segment_id,
            attention_mask=sentence_mask,
            output_all_encoded_layers=False,
        )

        paragraph_encoding = paragraph_encoding[:, qlen:, :]

        paragraph_encoding = self.fc_2(paragraph_encoding)

        return paragraph_encoding


class CrossPassageEncoder(PreTrainedBertModel):
    """
    When the individual chunk encodings are built from PassageEncoder, we assemble them all together to capture
    cross passage interactions.
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, vectors_in):
        # vectors_in is expected to be free of padding
        extended_attention_mask = torch.zeros(vectors_in.shape[0], vectors_in.shape[1],
                                              device=vectors_in.device, dtype=vectors_in.dtype)

        encoded_layers = self.encoder(
            vectors_in, extended_attention_mask, output_all_encoded_layers=False
        )
        encoded_layers = encoded_layers[-1]

        return encoded_layers


class SupportingFacts(nn.Module):
    def __init__(self, hypers: HypersSF, config: BertConfig, pretrained_state_dict=None):
        super().__init__()
        self.hypers = hypers
        self.passage_encoder = PassageEncoder(hypers, pretrained_state_dict=pretrained_state_dict)
        self.cross_passage_encoder = CrossPassageEncoder(config)
        if hypers.two_layer_sent_classifier:
            # try another layer here (the paper actually says there is another layer)
            # NOTE: this is just over budget on memory
            self.fc_sf = nn.Sequential(
                nn.Linear(hypers.transformer_hidden_size * 2, hypers.transformer_hidden_size),
                nn.ReLU(),
                nn.Linear(hypers.transformer_hidden_size, 1),
            )
        else:
            self.fc_sf = nn.Linear(hypers.transformer_hidden_size * 2, 1)

    def forward(self, batch):
        """
        Input is a tuple: (see dataloader)
        features.id, features.sent_ids, features.question_len, features.chunk_lengths,
               torch.tensor(chunk_token_ids, dtype=torch.long),
               torch.tensor(segment_ids, dtype=torch.long),
               torch.tensor(features.sent_starts, dtype=torch.long),
               torch.tensor(features.sent_ends, dtype=torch.long),
               sent_targets
        """
        id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, sent_targets = batch

        # encode each chunk (chunk = group of passages or sub-passage)
        paragraph_encodings = self.passage_encoder(chunk_tokens, segment_ids, qlen)
        assert len(paragraph_encodings.shape) == 3
        assert paragraph_encodings.shape[0] == len(chunk_lengths)

        # we've already trimmed the question from the front, now we trim the padding from the ends
        trimmed_paragraphs = []
        for ci in range(len(chunk_lengths)):
            trimmed_paragraphs.append(paragraph_encodings[ci, :chunk_lengths[ci], :])
        paragraph_encodings = torch.cat(trimmed_paragraphs, dim=0)

        # now get representations for passage tokens with attention over all passages
        all_tok_vecs = self.cross_passage_encoder(paragraph_encodings.view(1, -1, paragraph_encodings.shape[-1])).squeeze(0)
        assert len(all_tok_vecs.shape) == 2

        # get the [STARTSENT] and [ENDSENT] representations for each sentence
        start_vectors = all_tok_vecs[sent_starts]
        end_vectors = all_tok_vecs[sent_ends]
        start_end_concatenated = torch.cat([start_vectors, end_vectors], dim=-1)

        # classify each sentence as a supporting fact or not
        sentence_scores = self.fc_sf(start_end_concatenated).view(-1)
        sent_probs = torch.sigmoid(sentence_scores)

        # compute loss if ground truth is provided, otherwise give predictions
        if sent_targets is not None:
            loss_fct = torch.nn.BCELoss()
            if self.hypers.fp16:
                sent_targets = sent_targets.half()
            return loss_fct(sent_probs, sent_targets)
        else:
            return sent_probs

