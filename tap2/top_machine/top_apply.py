import logging
import os
import pprint
import random
import torch
import json

from pytorch_pretrained_bert.tokenization_offsets import BertTokenizer
from pytorch_pretrained_bert.hypers_rc import HypersRC
from pytorch_pretrained_bert.bert_trainer_apex import BertTrainer
from pytorch_pretrained_bert.dataloader.rc_data import RCData
from util.io_help import to_serializable

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def final_merger(ans_conf_list):
    # base implementation: just take the most confident answer from any source
    ans_conf_list.sort(key=lambda tup: tup[1], reverse=True)
    return ans_conf_list[0][0]


def validate(hypers, model, dev_dataset: RCData, prediction_file):
    model.eval()
    # need to take max confidence over the split questions
    qid2answer_conf = dict()
    for batch in dev_dataset.get_all_batches(0):
        batch = tuple(t.to(hypers.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, input_mask, segment_ids, _, _, _, _, qids, passage_texts, batch_token_offsets = batch
        logits = model(input_ids, segment_ids, input_mask, None, None, None, None)
        assert len(logits) == 3
        start_logits, end_logits, answer_type_logits = tuple(l.detach().cpu().numpy() for l in logits)
        assert start_logits.shape == end_logits.shape
        assert answer_type_logits.shape[1] == 3
        assert len(answer_type_logits.shape) == len(start_logits.shape) == 2
        for bi in range(len(qids)):
            qid = qids[bi]
            passage_text = passage_texts[bi]
            token_offsets = batch_token_offsets[bi]
            atl = answer_type_logits[bi]
            sl = start_logits[bi]
            el = end_logits[bi]
            scored_answers = RCData.get_predicted_answers(atl, sl, el, passage_text, token_offsets,
                                                          top_k=10, max_answer_length=hypers.max_answer_length)
            qid2answer_conf.setdefault(qid, []).extend(scored_answers)
            best_answer, conf = scored_answers[0]
            if random.random() < 0.001:
                logger.info(f'for {qid} predicted "{best_answer}" with score {conf}')

    os.makedirs(os.path.split(prediction_file)[0], exist_ok=True)
    with open(prediction_file, 'w') as f:
        json.dump(qid2answer_conf, f, default=to_serializable)


def main():
    parser = BertTrainer.get_base_parser()

    # Other parameters
    parser.add_argument("--load_model", default=None, type=str, required=False,
                        help="Bert model checkpoint file")
    parser.add_argument("--dev_file", default=None, type=str, help="RCData jsonl format")
    parser.add_argument("--cache_dir", default=None, type=str, help="Where to cache dataset features")
    parser.add_argument("--prediction_file", default=None, type=str,
                        help="Where to write the json predictions")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")

    args = parser.parse_args()
    hypers = HypersRC(args)

    logger.info(pprint.pformat(vars(hypers), indent=4))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.bert_model.endswith('uncased'))

    logger.info('resuming from saved model %s', args.load_model)
    checkpoint = torch.load(args.load_model, map_location='cpu')

    dev_data = RCData(args.dev_file, os.path.join(args.cache_dir, 'dev.pkl') if args.cache_dir else None,
                      tokenizer,
                      hypers.global_rank, hypers.world_size,
                      hypers.max_seq_length, hypers.doc_stride, hypers.max_query_length, hypers.train_batch_size,
                      first_answer_only=hypers.first_answer_only, fp16=hypers.fp16, include_source_info=True)

    # apply only
    model = BertTrainer.get_model(hypers, hypers.model_name, checkpoint, hypers_rc=hypers)
    validate(hypers, model, dev_data, args.prediction_file)


if __name__ == "__main__":
    main()
