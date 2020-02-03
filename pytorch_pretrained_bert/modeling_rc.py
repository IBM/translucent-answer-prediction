import logging
import torch
from torch import nn
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.dataloader.rc_data import AnswerType
from pytorch_pretrained_bert.hypers_rc import HypersRC

logger = logging.getLogger(__name__)


class BertForQuestionMultiAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    This version also uses the pooled_output to classify as no-answer, extracted-answer, yes, no, etc.
    non extracted-answer is indicated by 0,0 in the start and end positions
    the answer category is provided by the answer_category input Tensor

    This version also supports multiple allowed extracted answers, in case an answer string occurs more than once in the passage

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions,
            averaged with the answer-type loss if there is an answer-type classification component.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits, pool_logits. start and end logits are the logits respectively
            for the start and end position tokens of shape [batch_size, sequence_length].
            pool_logits has the logit of the probability that the passage has an answer of a certain type,
            shape [batch_size x answer-types]

    Example usage:

    """
    def __init__(self, config, hypers_rc: HypersRC=None):
        super(BertForQuestionMultiAnswering, self).__init__(config)
        self.bert = BertModel(config, no_pooler=(hypers_rc.num_answer_categories is None))
        self.num_categories = hypers_rc.num_answer_categories
        self.fp16 = hypers_rc.fp16

        if hypers_rc.two_layer_span_predict:
            self.qa_outputs = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 2),
            )
        else:
            self.qa_outputs = nn.Linear(config.hidden_size, 2)

        if hypers_rc.num_answer_categories is not None:
            logger.info(f'Including answer-type classifier with {hypers_rc.num_answer_categories} answer types')
            self.classifier = nn.Linear(config.hidden_size, hypers_rc.num_answer_categories, bias=False)
            # TM self.classifier = nn.Linear(config.hidden_size, num_categories)
        else:
            self.classifier = None
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, answer_mask=None,
                answer_type=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        if self.classifier is not None:
            pool_logits = self.classifier(pooled_output)  # TM self.dropout(pooled_output)
        else:
            pool_logits = None

        span_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # now batch_size x seq_length
        end_logits = end_logits.squeeze(-1)      # now batch_size x seq_length

        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            batch_size = start_positions.size(0)
            max_seq_length = input_ids.size(1)
            answer_options = start_positions.size(1)

            start_logits = start_logits.unsqueeze(-1).expand(batch_size, max_seq_length, answer_options)
            end_logits = end_logits.unsqueeze(-1).expand(batch_size, max_seq_length, answer_options)

            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=ignored_index)
            per_ans_loss = loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions) + answer_mask
            loss = per_ans_loss.min(1)[0]
            # mask out the loss for answer_types that are not span, CONSIDER: make this optional
            if answer_type is not None:
                # loss = torch.masked_select(loss, answer_type == AnswerType.span.value)
                answer_type_mask = (answer_type == AnswerType.span.value).to(dtype=loss.dtype)
                loss = loss * answer_type_mask
            loss = loss.mean()

            if pool_logits is not None:
                loss_fct = CrossEntropyLoss()
                category_loss = loss_fct(pool_logits.view(-1, self.num_categories), answer_type)
                loss = (loss + category_loss) / 2

            return loss
        else:
            if pool_logits is not None:
                return start_logits, end_logits, pool_logits
            else:
                return start_logits, end_logits
