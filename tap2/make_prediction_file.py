import logging
import argparse
import json
import os
import numpy as np
from evaluation.hotpotqa_evaluate_v1 import eval as hp_eval, exact_match_score, f1_score
from tap2.bottom_machine.tune_sf_thresholds import make_qid2sid2score
from util.args_help import file_list

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def confusion_matrix(answer_predictions, gt_file):
    def tondx(ans):
        if 'yes' == ans.lower():
            ndx = 0
        elif 'no' == ans.lower():
            ndx = 1
        else:
            ndx = 2
        return ndx
    con_matrix = np.zeros((3, 3), dtype=np.int32)
    qid2gt = dict()
    with open(gt_file, 'r') as fp:
        gt_json = json.load(fp)
    for q in gt_json:
        qid2gt[q['_id']] = q['answer']
    for id, predicted in answer_predictions.items():
        gold = qid2gt[id]
        gndx = tondx(gold)
        pndx = tondx(predicted)
        con_matrix[gndx, pndx] += 1
    return con_matrix


def metrics_by_full_recall(answer_predictions, qid2sid2score, threshold, gt_file):
    qid2sfs = dict()
    qid2gt = dict()
    with open(gt_file, 'r') as fp:
        jdata = json.load(fp)
    for jex in jdata:
        qid = jex['_id']
        supporting_facts = [sp[0] + ':' + str(sp[1]) for sp in jex['supporting_facts']]
        qid2sfs[qid] = supporting_facts
        qid2gt[qid] = jex['answer']

    sum_em_full_recall = 0
    sum_f1_full_recall = 0
    full_recall_count = 0
    sum_em_part_recall = 0
    sum_f1_part_recall = 0
    part_recall_count = 0
    for id, predicted in answer_predictions.items():
        gold = qid2gt[id]
        em = float(exact_match_score(predicted, gold))
        f1, _, _ = f1_score(predicted, gold)
        psf = [sid for sid, score in qid2sid2score[id].items() if score >= threshold]
        if all([q in psf for q in qid2sfs[id]]):
            full_recall_count += 1
            sum_em_full_recall += em
            sum_f1_full_recall += f1
        else:
            part_recall_count += 1
            sum_em_part_recall += em
            sum_f1_part_recall += f1
    logger.info(f'At threshold {threshold}')
    logger.info(f'Full Recall Count {full_recall_count}, '
                f'F1 = {sum_f1_full_recall/full_recall_count}, EM = {sum_em_full_recall/full_recall_count}')
    logger.info(f'Part Recall Count {part_recall_count}, '
                f'F1 = {sum_f1_part_recall/part_recall_count}, EM = {sum_em_part_recall/part_recall_count}')

def select_supporting_facts(scored_sfs, min_thresholds):
    if isinstance(scored_sfs, dict):
        scored_sfs = [(sf, score) for sf, score in scored_sfs.items()]
    scored_sfs.sort(key=lambda tup: tup[1], reverse=True)
    sfs = []
    for r, sf_score in enumerate(scored_sfs):
        sf, score = sf_score
        thresh = min_thresholds[min(r, len(min_thresholds)-1)]
        if score >= thresh:
            colon_ndx = sf.rindex(':')
            sfs.append((sf[:colon_ndx], int(sf[colon_ndx+1:])))
    return sfs


def ensemble_merger_inv_rank(multi_answer_confs):
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
    return scored_answers[0][0]


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
    return scored_answers[0][0]


def main():
    """
    convert the prediction and score files of our top and bottom machines to the HotpotQA prediction format

    bsingle make_prediction_file.py \
    --facts TAP/bm/large_out/plain_predictions.tsv,TAP/bm/large_out/sspt_predictions.tsv \
    --thresholds 0.45 \
    --answers TAP/tm/large_out/sspt_no_pool_predictions.json,TAP/tm/large_out/sspt_i_predictions.json \
    --data hotpot_dev_distractor_v1.json \
    --output TAP/predictions.json
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--facts", default=None, type=str, required=True,
                        help="From bottom_machine predictions.tsv")
    parser.add_argument("--answers", default=None, type=str, required=True,
                        help="From top_machine predictions.json, may be comma separated list")
    parser.add_argument("--thresholds", default=None, type=str, required=True,
                        help="Comma separated list of thresholds per-rank. Ex: 0,0.2,0.38")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Output file in HotpotQA format")
    parser.add_argument("--data", default=None, type=str, required=False,
                        help="HotpotQA json dataset, if present we run the eval script on the predictions")
    parser.add_argument("--qid2sid2bonus", default=None, type=str, required=False,
                        help="From boost_helpful_sfs")

    args = parser.parse_args()

    # min_thresholds = [0.38]  # thresholds for supporting facts, per-rank
    min_thresholds = [float(x) for x in args.thresholds.split(',')]

    qid2sfs = make_qid2sid2score(file_list(args.facts))

    if args.qid2sid2bonus:
        with open(args.qid2sid2bonus, 'r') as fp:
            qid2sid2bonus = json.load(fp)
        for qid, sid2bonus in qid2sid2bonus.items():
            sid2score = qid2sfs[qid]
            for sid, bonus in sid2bonus.items():
                # this has a tiny impact (maybe a tenth of a percent)
                sid2score[sid] = sid2score[sid] + bonus

    qid2multi_answer_confs = dict()
    if args.answers:
        for answer_file in file_list(args.answers):
            with open(answer_file, 'r') as fp:
                qid2answer_confs = json.load(fp)
                for qid, answer_confs in qid2answer_confs.items():
                    qid2multi_answer_confs.setdefault(qid, []).append(answer_confs)

    if not args.facts:
        qid2sfs = {qid: [] for qid in qid2multi_answer_confs.keys()}
    if not args.answers:
        qid2multi_answer_confs = {qid: [('abcxyz', 0.01)] for qid in qid2sfs.keys()}

    sf_predictions = dict()  # qid to list of (passage_id, sentence_number)
    for qid, sfs in qid2sfs.items():
        sf_predictions[qid] = select_supporting_facts(sfs, min_thresholds)

    answer_predictions = dict()  # qid to best answer string
    for qid, multi_answer_confs in qid2multi_answer_confs.items():
        answer_predictions[qid] = ensemble_merger_inv_rank(multi_answer_confs)
        # answer_predictions[qid] = ensemble_merger_mean_conf(multi_answer_confs)

    outdir = os.path.split(args.output)[0]
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, 'w') as fp:
        predictions = dict()
        predictions['sp'] = sf_predictions
        predictions['answer'] = answer_predictions
        json.dump(predictions, fp)

    if args.data:
        metrics = hp_eval(args.output, args.data)
        logger.info(f'metrics = {str(metrics)}')
        logger.info(f'confusion = \n{str(confusion_matrix(answer_predictions, args.data))}')
        metrics_by_full_recall(answer_predictions, qid2sfs, 0.1, args.data)


if __name__ == "__main__":
    main()
