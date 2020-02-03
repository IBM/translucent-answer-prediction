import logging
import argparse
import json

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# bsingle remove_ground_truth.py --data_limit 10 --data hotpot_dev_distractor_v1.json  --output input10.json
# bsingle remove_ground_truth.py --data hotpot_dev_distractor_v1.json  --output input.json
def main():
    """
    from a HotpotQA file, remove the ground truth (to test full_apply.sh)
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default=None, type=str, required=True,
                        help="HotpotQA json dataset")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Output file HotpotQA json dataset with ground truth removed")
    parser.add_argument("--data_limit", default=0, type=int, required=False,
                        help="limit on amount of data to load")

    args = parser.parse_args()

    with open(args.data, 'r') as fp:
        jdata = json.load(fp)

    for jex in jdata:
        del jex['supporting_facts']
        del jex['answer']

    if args.data_limit > 0:
        jdata = jdata[:args.data_limit]

    with open(args.output, 'w') as out:
        json.dump(jdata, out)


if __name__ == "__main__":
    main()
