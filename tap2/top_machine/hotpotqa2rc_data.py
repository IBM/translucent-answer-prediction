import logging
import argparse
import json
import random
from pytorch_pretrained_bert.dataloader.rc_data import AnswerType

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# python hotpotqa2rc_data.py --data hotpot_train_v1.1.json  --output train_rc_data.jsonl
# python hotpotqa2rc_data.py --data hotpot_dev_distractor_v1.json  --output dev_rc_data.jsonl
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
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Output file for jsonl RC data format")

    args = parser.parse_args()

    num_shuffles = 1  # the number of times each question is duplicated with different passage shuffles

    # NOTE: only appropriate for train
    with open(args.data, 'r') as fp:
        jdata = json.load(fp)
    with open(args.output, 'w') as out:
        for jex in jdata:
            qid = jex['_id']
            question = jex['question']
            supporting_facts = [sp[0] + ':' + str(sp[1]) for sp in jex['supporting_facts']]
            contexts = jex['context']
            answer = jex['answer']
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


if __name__ == "__main__":
    main()
