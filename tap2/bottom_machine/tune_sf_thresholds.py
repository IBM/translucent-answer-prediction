import logging
import argparse
import json
import numpy as np
from tap2.bottom_machine.confidence_reestimation import validate_score, fraction_full_recall, sf_stats
from util.args_help import file_list

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def make_sid2score(scored_sfs):
    sids = list(set([t[0] for t in scored_sfs]))
    sid2score = dict()
    for sid in sids:
        merge_score = np.array([t[1] for t in scored_sfs if t[0] == sid]).mean()  # TODO: try max
        sid2score[sid] = merge_score
    # TODO: also support rank bonuses
    return sid2score


def make_qid2sid2score(prediction_file_list):
    """
    first gather qid2sf_scores where each sf can have multiple score
    then final merger creating condensing the scores for each qid2sid2score
    then select a threshold to maximize F1
    :param prediction_file_list:
    :return:
    """
    qid2sf_scores = dict()
    for pfile in prediction_file_list:
        with open(pfile, 'r') as fp:
            for line in fp:
                parts = line.split('\t')
                if len(parts) != 3:
                    raise ValueError('bad line: '+line)
                qid = parts[0]
                sid = parts[1]
                score = float(parts[2])
                qid2sf_scores.setdefault(qid, []).append((sid, score))

    qid2sid2score = dict()
    for qid, sf_scores in qid2sf_scores.items():
        qid2sid2score[qid] = make_sid2score(sf_scores)
    return qid2sid2score


def main():
    """
    python tune_sf_thresholds.py \
    --predictions TAP/bm/large_out/plain_predictions.tsv,TAP/bm/large_out/sspt_predictions.tsv \
    --data hotpot_dev_distractor_v1.json \
    --qid2sid2bonus TAP/qid2sid2bonus.json
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=None, type=str, required=True,
                        help="HotpotQA json dataset")
    parser.add_argument("--predictions", default=None, type=str, required=True,
                        help="From bottom_machine predictions.tsv, comma separated")
    parser.add_argument("--qid2sid2bonus", default=None, type=str, required=False,
                        help="From boost_helpful_sfs")
    parser.add_argument("--threshold_plot", default=None, type=str, required=False,
                        help="Write tsv performance vs. threshold")
    args = parser.parse_args()

    qid2sid2score = make_qid2sid2score(file_list(args.predictions))

    if args.qid2sid2bonus:
        with open(args.qid2sid2bonus, 'r') as fp:
            qid2sid2bonus = json.load(fp)
        for qid, sid2bonus in qid2sid2bonus.items():
            sid2score = qid2sid2score[qid]
            for sid, bonus in sid2bonus.items():
                # this has a tiny impact (maybe a tenth of a percent)
                sid2score[sid] = sid2score[sid] + bonus

    qid2sfs = dict()
    with open(args.data, 'r') as fp:
        jdata = json.load(fp)
    for jex in jdata:
        qid = jex['_id']
        supporting_facts = [sp[0] + ':' + str(sp[1]) for sp in jex['supporting_facts']]
        qid2sfs[qid] = supporting_facts

    max_f1, max_em, best_thresh, scores = validate_score(qid2sfs, qid2sid2score)
    logger.info(f'other thresholds:')
    scores.sort(key=lambda t: t[2], reverse=True)  # sort by descending recall
    for f1, p, r, em, thresh in scores:
        logger.info(f'  F1 = {f1}, P = {p}, R = {r}, Threshold = {thresh}')
    # plot f, p, r, em as function of thresh
    scores.sort(key=lambda t: t[4])
    if args.threshold_plot:
        with open(args.threshold_plot, 'w') as f:
            for f1, p, r, em, thresh in scores:
                f.write(f'{thresh}\t{f1}\t{p}\t{r}\t{em}\n')
    fraction_full_recall(qid2sfs, qid2sid2score)
    sf_stats(qid2sid2score)


if __name__ == "__main__":
    main()
