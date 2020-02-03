import logging
import argparse
import json
import os
from evaluation.hotpotqa_evaluate_v1 import normalize_answer
from util.args_help import file_list

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_sid2text(data_file):
    """
    get sentence id to text of sentence dict
    :param data_file:
    :return:
    """
    with open(data_file, 'r') as fp:
        jdata = json.load(fp)
    sid2text = dict()
    for jex in jdata:
        qid = jex['_id']
        contexts = jex['context']
        for context in contexts:
            pid = context[0]
            sents = context[1]
            for ndx, sent in enumerate(sents):
                sid = pid + ":" + str(ndx)
                sid2text[sid] = sent
    return sid2text


def ensemble_merger_inv_rank(multi_answer_confs):
    """
    return the answer that has the highest sum-inverse-rank
    :param multi_answer_confs: a list of list of (answer, score) pairs
    :return:
    """
    answer_to_inv_ranks = dict()
    for answer_confs in multi_answer_confs:
        answer_confs.sort(key=lambda t: t[1], reverse=True)
        for ndx, answer_conf in enumerate(answer_confs):
            answer = answer_conf[0]
            norm_answer = answer.lower()
            answer_to_inv_ranks.setdefault(norm_answer, (answer, []))[1].append(1.0/(ndx+1))
    scored_answers = []
    for answer, inv_ranks in answer_to_inv_ranks.values():
        scored_answers.append((answer, sum(inv_ranks)))
    scored_answers.sort(key=lambda t: t[1], reverse=True)
    return scored_answers[0]


def ensemble_merger_mean_conf(multi_answer_confs):
    answer_to_scores = dict()
    for answer_confs in multi_answer_confs:
        answer_confs.sort(key=lambda t: t[1], reverse=True)
        for answer, conf in answer_confs:
            norm_answer = answer.lower()
            answer_to_scores.setdefault(norm_answer, (answer, []))[1].append(conf)
    scored_answers = []
    for answer, scores in answer_to_scores.values():
        scored_answers.append((answer, sum(scores)/len(scores)))
    scored_answers.sort(key=lambda t: t[1], reverse=True)
    return scored_answers[0]


def predicted_answer_bearing(ans, text):
    """
    does the answer appear in the text
    :param ans:
    :param text:
    :return:
    """
    return normalize_answer(ans) in normalize_answer(text)


def main():
    """
    If a question has an answer in the top 75% of confidences, boost the supporting fact score for sentences that
    contain that answer by 0.1

    python boost_helpful_sfs.py \
    --qid2sfs_used TAP/dev_pred_95_qid2sfs_used.json \
    --answers TAP/tm/large_out/sspt_i_95_predictions.json,TAP/tm/large_out/sspt_95_predictions.json \
    --data hotpot_dev_distractor_v1.json \
    --output TAP/qid2sid2bonus.json
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--answers", default=None, type=str, required=True,
                        help="From top_machine predictions.json, may be comma separated list")
    parser.add_argument("--qid2sfs_used", default=None, type=str, required=True,
                        help="Output file for json qid to supporting facts used")
    parser.add_argument("--data", default=None, type=str, required=True,
                        help="HotpotQA json dataset, we use it to find which sentences are predicted-answer-bearing")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Output file in HotpotQA format")

    args = parser.parse_args()
    # CONSIDER: different bonuses for different confidences?
    confidence_percentile = 0.75
    high_conf_bonus = 0.0
    high_conf_ans_bearing_bonus = 0.1

    qid2multi_answer_confs = dict()
    for answer_file in file_list(args.answers):
        with open(answer_file, 'r') as fp:
            qid2answer_confs = json.load(fp)
            for qid, answer_confs in qid2answer_confs.items():
                qid2multi_answer_confs.setdefault(qid, []).append(answer_confs)
    answer_predictions = dict()  # qid to best answer string and score
    for qid, multi_answer_confs in qid2multi_answer_confs.items():
        answer_predictions[qid] = ensemble_merger_inv_rank(multi_answer_confs)
        # answer_predictions[qid] = ensemble_merger_mean_conf(multi_answer_confs)

    # find distribution of confidences to define 'high confidence'
    best_confs = [conf for ans, conf in answer_predictions.values()]
    best_confs.sort()
    high_conf = best_confs[int(len(best_confs) * confidence_percentile)]  # top X% of confidences
    logger.info(f'high confidence threshold = {high_conf}')

    with open(args.qid2sfs_used, 'r') as fp:
        qid2sfs_used = json.load(fp)
    all_sids_used = set()
    for qid, sfs_used in qid2sfs_used.items():
        for sfu in sfs_used:
            all_sids_used.add(sfu)

    sid2text = get_sid2text(args.data)

    qid2sid2bonus = dict()
    for qid, sfs_used in qid2sfs_used.items():
        ans, score = answer_predictions[qid]
        if score < high_conf:
            continue
        sid2bonus = dict()
        for sid in sfs_used:
            if predicted_answer_bearing(ans, sid2text[sid]):
                sid2bonus[sid] = high_conf_ans_bearing_bonus
            else:
                sid2bonus[sid] = high_conf_bonus
        qid2sid2bonus[qid] = sid2bonus

    outdir = os.path.split(args.output)[0]
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, 'w') as fp:
        json.dump(qid2sid2bonus, fp)


if __name__ == "__main__":
    main()
