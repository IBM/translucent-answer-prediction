import logging
import os
import json
from datetime import datetime

from pytorch_pretrained_bert.bert_trainer_apex import Hypers

logger = logging.getLogger(__name__)


class HypersRC(Hypers):
    def __init__(self, args):
        super().__init__(args)
        self.two_layer_span_predict = True
        self.max_seq_length = args.max_seq_length
        self.doc_stride = args.doc_stride
        self.max_query_length = args.max_query_length
        self.model_name = 'pytorch_pretrained_bert.modeling_rc.BertForQuestionMultiAnswering'
        self.num_answer_categories = 3
        self.learning_rate = 3e-5
        # maximum length considered for an answer span, in wordpieces
        self.max_answer_length = 15
        self.first_answer_only = args.first_answer_only if hasattr(args, 'first_answer_only') else False
        # the name of the f1 scoring function for validation in train_rc
        self.scoring_impl = 'HotpotQA'
        if hasattr(args, 'experiment_name') and args.experiment_name:
            self.experiment_name = args.experiment_name
        else:
            self.experiment_name = f'rc_{str(datetime.now())}'
        self.report_freq = 200

    def write_results_file(self, results_dir, **kwargs):
        os.makedirs(results_dir, exist_ok=True)
        file = os.path.join(results_dir, self.experiment_name+'.json')
        run_num = 1
        while os.path.exists(file):
            run_num += 1
            file = os.path.join(results_dir, self.experiment_name+f'_{run_num}.json')
        result_dict = dict(self.__dict__)
        del result_dict['device']  # not JSON serializable and not really a hyperparameter
        del result_dict['local_rank']
        del result_dict['global_rank']
        # add git information about the version of the code
        if 'GIT_REPOSITORY' in os.environ:
            result_dict['git_repository'] = os.environ['GIT_REPOSITORY']
        if 'GIT_COMMIT_ID' in os.environ:
            result_dict['git_commit_id'] = os.environ['GIT_COMMIT_ID']
        # add extra info (results, instances-per-second)
        for k, v in kwargs.items():
            result_dict[k] = v
        with open(file, 'w') as fp:
            json.dump(result_dict, fp, indent=4)
