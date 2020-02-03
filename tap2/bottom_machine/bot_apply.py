import torch
import time
import os

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertConfig

from tap2.bottom_machine.dataloader import get_data
from tap2.bottom_machine.model_sf import SupportingFacts
from tap2.bottom_machine.hypers import HypersSF
import logging
from util.reporting import Reporting
from util.io_help import cached_load

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_to_device(batch, device):
    return tuple((t.to(device) if isinstance(t, torch.Tensor) else t) for t in batch)


def write_prediction_file(qid2sid2score, file):
    os.makedirs(os.path.split(file)[0], exist_ok=True)
    with open(file, 'w') as f:
        for qid, sid2score in qid2sid2score.items():
            for sid, score in sid2score.items():
                f.write(f'{qid}\t{sid}\t{score}\n')


def validate(hypers, model, dev_dataset, prediction_file):
    model.eval()
    qid2sid2score = dict()
    report = Reporting()
    for bn in range(len(dev_dataset)):
        if report.is_time():
            logger.info(f'Apply on bottom instance {report.check_count}')
        batch = dev_dataset[bn]
        id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, sent_targets = batch_to_device(batch, hypers.device)
        sent_probs = model((id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, None))
        sent_probs = sent_probs.detach().cpu().numpy()
        sid2score = qid2sid2score.setdefault(id, dict())
        for sid, score in zip(sent_ids, sent_probs):
            if sid not in sid2score or sid2score[sid] < score:
                sid2score[sid] = score
    write_prediction_file(qid2sid2score, prediction_file)


def main():
    parser = HypersSF.get_base_parser()

    parser.add_argument("--dev_data", default=None, type=str, required=False,
                        help="HotpotQA json dataset for dev")
    parser.add_argument("--cache_dir", default=None, type=str, required=False,
                        help="Directory to save the cached dataset pickles")
    parser.add_argument("--data_limit", default=0, type=int, required=False,
                        help="limit on amount of data to load")
    parser.add_argument("--load_model", default=None, type=str, required=False,
                        help="File to load SF model from")
    parser.add_argument("--prediction_file", default=None, type=str, required=False,
                        help="Write the supporting fact predictions, with confidence to this file")
    parser.add_argument("--no_blank", default=False, action='store_true',
                        help="Do not use [BLANK], for old models.")

    args = parser.parse_args()
    hypers = HypersSF(args)

    special_tokens = ['[BLANK]', '[STARTSENT]', '[ENDSENT]']
    if args.no_blank:
        special_tokens = ['[STARTSENT]', '[ENDSENT]']
    # [BLANK] so if we use the SSPT model here then [STARTSENT] and [ENDSENT] are still two *new* special tokens
    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.bert_model.endswith('uncased'),
                                              add_special_tokens=special_tokens)

    logger.info(f'vocab size = {len(tokenizer.vocab)}')
    transformer_config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab),
                                    hidden_size=hypers.transformer_hidden_size,
                                    num_hidden_layers=hypers.num_transformer_layers,
                                    num_attention_heads=8, intermediate_size=2048)  # TODO: should these be hyperparameters

    # create model and wrap with DistributedDataParallel
    model_sf = SupportingFacts(hypers, transformer_config, pretrained_state_dict=None)
    model_sf.load_state_dict(torch.load(args.load_model, map_location=hypers.device))
    model = hypers.wrap_model(model_sf)

    # load data
    start_time = time.time()
    dev_dataset, _ = \
        cached_load(os.path.join(args.cache_dir, 'dev_dataset.pkl') if args.cache_dir else None,
                    lambda: get_data(args.dev_data, tokenizer, hypers.data_options, limit=args.data_limit),
                    should_save=(hypers.global_rank == 0))
    logger.info(f'loaded datasets in {time.time()-start_time} seconds')

    validate(hypers, model, dev_dataset, args.prediction_file)


if __name__ == "__main__":
    main()
