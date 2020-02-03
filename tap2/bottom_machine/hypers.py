from torch_util.hypers_base import HypersBase
from enum import Enum, unique


@unique
class SentMarkerStyle(Enum):
    """
    When building representations for sentences in supporting fact prediction, how do we get the sentence representation
    """
    rep = 0,   # add start and end markers and use them for the sentence representation
    mark = 1,  # add start and end markers but use the first and last real token for the representation
    no = 2     # do not use start and end markers


class DataOptions():
    """
    Options that control data preprocessing for supporting fact prediction.
    Changing these requires rebuilding the cached files to take effect.
    """
    def __init__(self):
        self.max_seq_len = 512
        # whether to truncate or split passages that are too long
        self.truncate_passages = True
        # if the passages don't fit into num_para_chunks, do we add another example for the question?
        self.split_questions = False
        # how many blocks of max_seq_len do we process per-question (at most)
        self.num_para_chunks = 4
        # truncate questions longer than this
        self.max_question_len = 35
        # do we add [SENTSTART] and [SENTEND] to each sentence? and do we get the representation from them?
        self.sent_marker_style = SentMarkerStyle.no


class HypersSF(HypersBase):
    def __init__(self, args):
        super().__init__(args)
        self.data_options = DataOptions()
        self.epochs = 4
        self.bert_model = args.bert_model
        if self.bert_model.startswith("bert-base"):
            self.bert_hidden_size = 768
        else:
            self.bert_hidden_size = 1024
        self.num_transformer_layers = 2
        self.transformer_hidden_size = 512
        self.dropout = 0.1  # doesn't apply to BERT
        self.bert_archive = None
        self.data_options.sent_marker_style = SentMarkerStyle[args.sent_marker_style]
        self.two_layer_sent_classifier = args.two_layer_sent_classifier if hasattr(args, 'two_layer_sent_classifier') else False

    @classmethod
    def get_base_parser(cls):
        parser = super().get_base_parser()
        parser.add_argument("--bert_model", default=None, type=str, required=True,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
        return parser
