import logging
import os
import pprint
import time
import random
import torch
import numpy as np
import json

from pytorch_pretrained_bert.tokenization_offsets import BertTokenizer
from pytorch_pretrained_bert.hypers_rc import HypersRC
from pytorch_pretrained_bert.bert_trainer_apex import BertTrainer
from pytorch_pretrained_bert.dataloader.rc_data import RCData
from evaluation.hotpotqa_evaluate_v1 import f1_score as hp_f1_score
from util.io_help import to_serializable

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def f1_score(impl, gt_answer, predicted_answer):
    if impl == 'HotpotQA':
        return hp_f1_score(predicted_answer, gt_answer)[0]
    else:
        raise ValueError(f'unknown f1_score implementation: {impl}')


def match_score(impl, gt_answers, predicted_answer):
    return max([f1_score(impl, gt_answer, predicted_answer) for gt_answer in gt_answers])


def final_merger(ans_conf_list):
    # base implementation: just take the most confident answer from any source
    ans_conf_list.sort(key=lambda tup: tup[1], reverse=True)
    return ans_conf_list[0][0]


def validate(hypers, model, dev_dataset: RCData, prediction_file=None, best_f1=0):
    if hypers.global_rank != 0:
        return None
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
            # TODO: because we can have doc_stride, etc we need to weight the confidence by answer_type != noanswer
            scored_answers = RCData.get_predicted_answers(atl, sl, el, passage_text, token_offsets,
                                                          top_k=10, max_answer_length=hypers.max_answer_length)
            qid2answer_conf.setdefault(qid, []).extend(scored_answers)
            best_answer, conf = scored_answers[0]
            if random.random() < 0.001:
                logger.info(f'for {qid} predicted "{best_answer}" with score {conf}')
    sum_score = 0
    count = 0
    for qid, ans_conf_list in qid2answer_conf.items():
        best_answer = final_merger(ans_conf_list)
        qid_f1 = match_score(hypers.scoring_impl, dev_dataset.qid2gt[qid], best_answer)
        sum_score += qid_f1
        count += 1
    assert len(dev_dataset.qid2gt) == count

    f1_score = sum_score/count
    if prediction_file is not None and f1_score > best_f1:
        os.makedirs(os.path.split(prediction_file)[0], exist_ok=True)
        with open(prediction_file, 'w') as f:
            json.dump(qid2answer_conf, f, default=to_serializable)

    return f1_score


def validate_distributed(hypers, model, dev_dataset: RCData):
    model.eval()
    # need to take max confidence over the split questions
    qid2confidence_f1 = dict()
    # we can't all_gather qid strings, we need to convert the qids to ndxs
    all_qids = dev_dataset.all_qids()
    qid2ndx = {qid: ndx for ndx, qid in enumerate(all_qids)}
    for batch in dev_dataset.get_batches(0):
        batch = tuple(t.to(hypers.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, input_mask, segment_ids, _, _, _, _, qids, passage_texts, batch_token_offsets = batch
        logits = model(input_ids, segment_ids, input_mask, None, None, None, None)
        assert len(logits) == 3
        start_logits, end_logits, answer_type_logits = tuple(l.detach().cpu().numpy() for l in logits)
        assert start_logits.shape == end_logits.shape
        assert answer_type_logits.shape[1] == 3
        assert len(answer_type_logits.shape) == len(start_logits.shape) == 2
        qid_indices = torch.zeros(hypers.train_batch_size, dtype=torch.float32, device=hypers.device)
        answer_confs = torch.zeros(hypers.train_batch_size, dtype=torch.float32, device=hypers.device)
        answer_scores = torch.zeros(hypers.train_batch_size, dtype=torch.float32, device=hypers.device)
        # hypers.train_batch_size is actually an upper limit, we can have smaller batches so we pass a mask vector
        answer_mask = torch.zeros(hypers.train_batch_size, dtype=torch.float32, device=hypers.device)
        for bi in range(len(qids)):
            qid = qids[bi]
            passage_text = passage_texts[bi]
            token_offsets = batch_token_offsets[bi]
            atl = answer_type_logits[bi]
            sl = start_logits[bi]
            el = end_logits[bi]
            scored_answers = RCData.get_predicted_answers(atl, sl, el, passage_text, token_offsets,
                                                          top_k=1, max_answer_length=hypers.max_answer_length)
            best_answer, conf = scored_answers[0]
            if hypers.global_rank == 0 and random.random() < 0.01:
                logger.info(f'for {qid} predicted "{best_answer}" with score {conf}')
            qid_indices[bi] = float(qid2ndx[qid])  # just because the stacked tensor needs to all be float
            answer_confs[bi] = float(conf)
            answer_scores[bi] = match_score(hypers.scoring_impl, dev_dataset.qid2gt[qid], best_answer)
            answer_mask[bi] = 1
        results = torch.stack((answer_mask, qid_indices, answer_confs, answer_scores))
        gather_list = [torch.zeros_like(results) for _ in range(hypers.world_size)]
        torch.distributed.all_gather(gather_list, results)  # because NCCL does not support plain 'gather'
        if hypers.global_rank == 0:
            for i in range(len(gather_list)):
                qid_ndxs = torch.masked_select(gather_list[i][1, :], gather_list[i][0, :] == 1).cpu().numpy().astype(dtype=np.int32)
                confs = torch.masked_select(gather_list[i][2, :], gather_list[i][0, :] == 1).cpu().numpy()
                scores = torch.masked_select(gather_list[i][3, :], gather_list[i][0, :] == 1).cpu().numpy()
                for bi in range(len(qid_ndxs)):
                    qid2confidence_f1.setdefault(all_qids[qid_ndxs[bi]], []).append((confs[bi], scores[bi]))
    sum_score = 0
    count = 0
    if hypers.global_rank == 0:
        for qid, conf_f1_list in qid2confidence_f1.items():
            conf_f1_list.sort(reverse=True)
            sum_score += conf_f1_list[0][1]
            count += 1
        assert len(all_qids) == count
        return sum_score/count
    else:
        return None


def check_batch(batch):
    input_ids, input_mask, segment_ids, start_positions, end_positions, answer_mask, answer_type = batch
    assert input_ids is not None
    assert input_mask is not None
    assert segment_ids is not None
    assert start_positions is not None
    assert end_positions is not None
    assert answer_mask is not None
    assert answer_type is not None
    assert input_ids.shape[0] == input_mask.shape[0] == segment_ids.shape[0] == start_positions.shape[0] == \
           end_positions.shape[0] == answer_mask.shape[0] == answer_type.shape[0]
    assert input_ids.shape[1] == input_mask.shape[1] == segment_ids.shape[1]
    assert len(start_positions.shape) == len(end_positions.shape) == len(answer_mask.shape) == 2
    assert start_positions.shape[1] == end_positions.shape[1] == answer_mask.shape[1]


def main():
    parser = BertTrainer.get_base_parser()

    # Other parameters
    parser.add_argument("--save_model", default=None, type=str, required=False,
                        help="Bert model checkpoint file")
    parser.add_argument("--load_model", default=None, type=str, required=False,
                        help="Bert model checkpoint file")
    parser.add_argument("--first_answer_only", default=False, action='store_true',
                        help="the target answer span is only the first one")
    parser.add_argument("--train_file", default=None, type=str, help="RCData jsonl format")
    parser.add_argument("--dev_file", default=None, type=str, help="RCData jsonl format")
    parser.add_argument("--cache_dir", default=None, type=str, help="Where to cache dataset features")
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="The name of the experiment, to store in the results directory")
    parser.add_argument("--results_dir", default=None, type=str,
                        help="Where to store the json results with hyperparameters")
    parser.add_argument("--prediction_file", default=None, type=str,
                        help="Where to write the json predictions")
    parser.add_argument("--num_epochs", default=4, type=int,
                        help="Number of epochs to train for.")
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
    logger.info('torch cuda version %s', torch.version.cuda)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.bert_model.endswith('uncased'))

    checkpoint = None
    if args.load_model:
        logger.info('resuming from saved model %s', args.load_model)
        checkpoint = torch.load(args.load_model, map_location='cpu')
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            del checkpoint['optimizer']

    dev_data = RCData(args.dev_file, os.path.join(args.cache_dir, 'dev.pkl') if args.cache_dir else None,
                      tokenizer,
                      hypers.global_rank, hypers.world_size,
                      hypers.max_seq_length, hypers.doc_stride, hypers.max_query_length, hypers.train_batch_size,
                      first_answer_only=hypers.first_answer_only, fp16=hypers.fp16, include_source_info=True)

    # apply only
    if not args.train_file:
        if hypers.global_rank == 0:
            model = BertTrainer.get_model(hypers, hypers.model_name, checkpoint, hypers_rc=hypers)
            validate_start_time = time.time()
            f1 = validate(hypers, model, dev_data,
                          prediction_file=args.prediction_file,
                          best_f1=0)
            logger.info(f'F1 = {f1}')
            logger.info(f'Took {time.time()-validate_start_time} seconds')
            if args.results_dir:
                hypers.write_results_file(args.results_dir, f1=f1)
        return

    train_data = RCData(args.train_file, os.path.join(args.cache_dir, 'train.pkl') if args.cache_dir else None,
                        tokenizer,
                        hypers.global_rank, hypers.world_size,
                        hypers.max_seq_length, hypers.doc_stride, hypers.max_query_length, hypers.train_batch_size,
                        first_answer_only=hypers.first_answer_only, fp16=hypers.fp16)

    hypers.num_train_steps = args.num_epochs * train_data.num_batches/hypers.gradient_accumulation_steps

    # Prepare trainer
    trainer = BertTrainer(hypers, hypers.model_name, checkpoint, hypers_rc=hypers)

    epoch_f1s = []
    for epoch in range(args.num_epochs):
        trainer.reset()
        trainer.model.train()
        for batch in train_data.get_batches(epoch):
            if not trainer.should_continue():
                logger.error('num train steps calculated wrong!')
                break
            batch = tuple(t.to(hypers.device) if isinstance(t, torch.Tensor) else t for t in batch)
            # check_batch(batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions, answer_mask, answer_type = batch

            loss = trainer.model(input_ids, segment_ids, input_mask,
                                 start_positions, end_positions, answer_mask,
                                 answer_type)
            trainer.step_loss(loss)

        if hypers.global_rank == 0:
            validate_start_time = time.time()
            f1 = validate(hypers, trainer.model, dev_data,
                          prediction_file=args.prediction_file,
                          best_f1=max(epoch_f1s) if epoch_f1s else 0)
            logger.info(f'Epoch {epoch+1} F1 = {f1}')
            logger.info(f'Took {time.time()-validate_start_time} seconds')
            epoch_f1s.append(f1)
            if f1 == max(epoch_f1s):
                trainer.save_simple(args.save_model)

    if hypers.global_rank == 0:
        if args.results_dir:
            hypers.write_results_file(args.results_dir,
                                      instances_per_second=trainer.train_stats.instances_per_second(),
                                      f1s=epoch_f1s,
                                      f1=max(epoch_f1s))


if __name__ == "__main__":
    main()
