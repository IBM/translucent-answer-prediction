import logging
import argparse
import json
import random
from pytorch_pretrained_bert.dataloader.rc_data import AnswerType
from tap2.bottom_machine.tune_sf_thresholds import make_qid2sid2score
from util.args_help import file_list

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def select_supporting_facts(scored_sfs, min_thresholds):
    """
    select supporting facts according to the provided thresholds
    :param scored_sfs: a list of (sentence_id, score)
    :param min_thresholds: a list of minimum scores for top ranked supporting facts:
           [min_score_for_top_ranked, min_score_for_second_ranked, min_score_for_others]
    :return: a list of sentence ids predicted as supporting facts
    """
    if isinstance(scored_sfs, dict):
        scored_sfs = [(sf, score) for sf, score in scored_sfs.items()]
    scored_sfs.sort(key=lambda tup: tup[1], reverse=True)
    sfs = []
    for r, sf_score in enumerate(scored_sfs):
        sf, score = sf_score
        thresh = min_thresholds[min(r, len(min_thresholds)-1)]
        if score >= thresh:
            sfs.append(sf)
    return sfs


# python predictions2rc_data.py \
# --predictions TAP/bm/large_out/predictions.tsv \
# --data hotpot_dev_distractor_v1.json \
# --output TAP/dev_pred2_rc_data.jsonl
def main():
    """
    Convert the HotpotQA distractor setting dataset into the jsonl format used by rc_data.
    Each jsonl RC record has:
    qid : string id for this question
    question : question string
    passage : passage string
    answer_type : if present, the AnswerType (yes, no, span)
    answers : a list of acceptable answer strings
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=None, type=str, required=True,
                        help="HotpotQA json dataset")
    parser.add_argument("--predictions", default=None, type=str, required=True,
                        help="From bottom_machine predictions.tsv")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Output file for jsonl RC data format")
    parser.add_argument("--thresholds", default=None, type=str, required=True,
                        help="Comma separated list of thresholds per-rank. Ex: 0,0.2,0.38")
    parser.add_argument("--qid2sfs_used", default=None, type=str, required=False,
                        help="Output file for json qid to supporting facts used")

    args = parser.parse_args()

    # min_thresholds = [0.0, 0.2, 0.38]  # thresholds for supporting facts, per-rank
    min_thresholds = [float(x) for x in args.thresholds.split(',')]
    if min_thresholds[0] > 0:
        logger.warning('WARNING: non-zero threshold for top supporting fact prediction!')
    num_shuffles = 1  # the number of times each question is duplicated with different passage shuffles

    qid2sfs = make_qid2sid2score(file_list(args.predictions))

    with open(args.data, 'r') as fp:
        jdata = json.load(fp)

    qid2sfs_used = dict()
    with open(args.output, 'w') as out:
        for jex in jdata:
            qid = jex['_id']
            question = jex['question']
            contexts = jex['context']
            answer = jex['answer'] if 'answer' in jex else 'TEST QUESTION'
            supporting_facts = select_supporting_facts(qid2sfs[qid], min_thresholds)
            qid2sfs_used[qid] = supporting_facts
            supporting_fact_passages = []  # a list of passage substring that contain supporting facts
            for context in contexts:
                pid = context[0]
                sents = context[1]
                sfs_in_passage = ''
                for ndx, sent in enumerate(sents):
                    sid = pid + ":" + str(ndx)
                    if sid in supporting_facts:
                        sfs_in_passage += sent + '\n'
                if sfs_in_passage:
                    supporting_fact_passages.append(sfs_in_passage)
            passage = ''.join(supporting_fact_passages)
            if len(passage.strip()) == 0:
                logger.warning(f'bad passage selected: {passage} for {qid}')
                continue
            created_passages = set()
            for shuffle_i in range(num_shuffles):
                # the 'passage' we synthesize can have the supporting_fact_passages concatenated in any order
                for _ in range(10):
                    random.shuffle(supporting_fact_passages)
                    passage = ''.join(supporting_fact_passages)
                    if passage not in created_passages:
                        break
                rc_data_rec = dict()
                rc_data_rec['qid'] = qid
                rc_data_rec['question'] = question
                rc_data_rec['passage'] = passage
                if answer.lower() == 'yes':
                    rc_data_rec['answer_type'] = AnswerType.yes.name
                elif answer.lower() == 'no':
                    rc_data_rec['answer_type'] = AnswerType.no.name
                else:
                    rc_data_rec['answer_type'] = AnswerType.span.name
                rc_data_rec['answers'] = [answer]
                created_passages.add(passage)
                out.write(json.dumps(rc_data_rec)+'\n')

    # we record what sentences are selected for the composite passage so if we predict an answer with high confidence,
    # we increase the score for those supporting facts
    if args.qid2sfs_used:
        with open(args.qid2sfs_used, 'w') as fp:
            json.dump(qid2sfs_used, fp)


if __name__ == "__main__":
    main()
