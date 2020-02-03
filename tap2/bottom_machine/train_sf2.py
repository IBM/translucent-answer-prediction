import torch
import random
import time
import os
import math

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertConfig

from tap2.bottom_machine.dataloader import get_data
from tap2.bottom_machine.model_sf import SupportingFacts
from tap2.bottom_machine.hypers import HypersSF
from torch_util.optimizer import Optimizer
import logging
from util.reporting import Reporting
from tap2.bottom_machine.confidence_reestimation import validate_score
from util.io_help import cached_load

logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_to_device(batch):
    return tuple((t.to(torch.device("cuda")) if isinstance(t, torch.Tensor) else t) for t in batch)


def write_prediction_file(qid2sid2score, file):
    os.makedirs(os.path.split(file)[0], exist_ok=True)
    with open(file, 'w') as f:
        for qid, sid2score in qid2sid2score.items():
            for sid, score in sid2score.items():
                f.write(f'{qid}\t{sid}\t{score}\n')


def validate(hypers, model, dev_dataset, qid2sfs, prediction_file=None):
    model.eval()
    # assert set(f[0] for f in dev_dataset) == qid2sfs.keys()
    leftovers = len(dev_dataset) % hypers.world_size  # the ones that don't divide evenly for distributed
    max_sentences = 128  # the maximum number of sentences that could be in a single chunk
    # gather y_true, y_scores per qid (only on global_rank == 0)
    qid2sid2score = dict()
    for bn in range(leftovers, len(dev_dataset), hypers.world_size):
        batch = dev_dataset[bn+hypers.global_rank]
        id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, sent_targets = batch_to_device(batch)
        sent_probs = model((id, sent_ids, qlen,
                            chunk_lengths, chunk_tokens, segment_ids,
                            sent_starts, sent_ends, None)).detach()

        # pad sent_probs and sent_targets to the maximum number of sentences
        num_sents = sent_probs.shape[0]
        if num_sents > max_sentences:
            logger.error(f'too many sentences {num_sents}')
        assert num_sents <= max_sentences
        sent_probs_padded = torch.zeros(max_sentences, dtype=torch.float32).to(hypers.device)
        sent_probs_padded[:num_sents] = sent_probs
        mask = torch.zeros(max_sentences, dtype=torch.float32).to(hypers.device)
        mask[:num_sents] = 1
        # gather results over all nodes
        results = torch.stack((sent_probs_padded, mask))
        gather_list = [torch.zeros_like(results) for _ in range(hypers.world_size)]
        torch.distributed.all_gather(gather_list, results)  # because NCCL does not support plain 'gather'

        if hypers.global_rank == 0:
            ids = [b[0] for b in dev_dataset[bn:bn+hypers.world_size]]
            batch_sent_ids = [b[1] for b in dev_dataset[bn:bn+hypers.world_size]]
            for i in range(len(ids)):
                id = ids[i]
                sent_ids = batch_sent_ids[i]
                sent_probs = torch.masked_select(gather_list[i][0, :], gather_list[i][1, :] == 1).cpu().numpy()
                sid2score = qid2sid2score.setdefault(id, dict())
                for sid, score in zip(sent_ids, sent_probs):
                    if sid not in sid2score or sid2score[sid] < score:
                        sid2score[sid] = score

    if hypers.global_rank == 0:
        # run the leftovers on model.module on global_rank 0
        for bn in range(leftovers):
            batch = dev_dataset[bn]
            id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, sent_targets = batch_to_device(batch)
            sent_probs = model.module((id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, None))
            sent_probs = sent_probs.detach().cpu().numpy()
            sid2score = qid2sid2score.setdefault(id, dict())
            for sid, score in zip(sent_ids, sent_probs):
                if sid not in sid2score or sid2score[sid] < score:
                    sid2score[sid] = score

        # assert len(qid2sfs) == len(qid2sid2score)
        assert qid2sfs.keys() == qid2sid2score.keys()
        # compute scores
        if prediction_file is not None:
            write_prediction_file(qid2sid2score, prediction_file)
        return validate_score(qid2sfs, qid2sid2score)
    else:
        return None, None, None, None


def main():
    parser = HypersSF.get_base_parser()

    parser.add_argument("--train_data", default=None, type=str, required=False,
                        help="HotpotQA json dataset for train")
    parser.add_argument("--dev_data", default=None, type=str, required=False,
                        help="HotpotQA json dataset for dev")
    parser.add_argument("--cache_dir", default=None, type=str, required=False,
                        help="Directory to save the cached dataset pickles")
    parser.add_argument("--data_limit", default=0, type=int, required=False,
                        help="limit on amount of data to load")
    parser.add_argument("--save_model", default=None, type=str, required=False,
                        help="File to save SF model to")
    parser.add_argument("--load_model", default=None, type=str, required=False,
                        help="File to load SF model from")
    parser.add_argument("--pretrained_model", default=None, type=str, required=False,
                        help="File to load pre-trained BERT model from")
    parser.add_argument("--prediction_file", default=None, type=str, required=False,
                        help="Write the supporting fact predictions, with confidence to this file")
    parser.add_argument("--fold", default=-1, type=int, required=False,
                        help="if >= 0, we are doing five fold cross apply on train. folds will be in [0,5)")
    parser.add_argument("--no_validate", default=False, action='store_true',
                        help="Do not run validate, only train and save model. Used when fold is set.")
    parser.add_argument("--experiment_name", default=None, type=str,
                        help="The name of the experiment, to store in the results directory")
    parser.add_argument("--results_dir", default=None, type=str,
                        help="Where to store the json results with hyperparameters")
    parser.add_argument("--sent_marker_style", default="no", type=str,
                        help="Whether to introduce special tokens for sentence start and end, and how to use them")
    parser.add_argument("--two_layer_sent_classifier", default=False, action='store_true',
                        help="Use two FC layers for sentence classification.")

    args = parser.parse_args()
    hypers = HypersSF(args)

    # [BLANK] so if we use the SSPT model here then [STARTSENT] and [ENDSENT] are still two *new* special tokens
    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.bert_model.endswith('uncased'),
                                              add_special_tokens=['[BLANK]', '[STARTSENT]', '[ENDSENT]'])

    logger.info(f'vocab size = {len(tokenizer.vocab)}')
    transformer_config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab),
                                    hidden_size=hypers.transformer_hidden_size,
                                    num_hidden_layers=hypers.num_transformer_layers,
                                    num_attention_heads=8, intermediate_size=2048)  # TODO: should these be hyperparameters

    pretrained_state_dict = None
    if args.pretrained_model:
        pretrained_state_dict = torch.load(args.pretrained_model)
    # create model and wrap with DistributedDataParallel
    model_sf = SupportingFacts(hypers, transformer_config, pretrained_state_dict=pretrained_state_dict)
    if args.load_model:
        model_sf.load_state_dict(torch.load(args.load_model))
    model = hypers.wrap_model(model_sf)

    # load data
    start_time = time.time()
    if args.train_data:
        train_dataset, train_qid2sfs = \
            cached_load(os.path.join(args.cache_dir, 'train_dataset.pkl') if args.cache_dir else None,
                        lambda: get_data(args.train_data, tokenizer, hypers.data_options, limit=args.data_limit),
                        should_save=(hypers.global_rank == 0))
    else:
        train_dataset, train_qid2sfs = None, None
    if args.dev_data:
        dev_dataset, qid2sfs = \
            cached_load(os.path.join(args.cache_dir, 'dev_dataset.pkl') if args.cache_dir else None,
                        lambda: get_data(args.dev_data, tokenizer, hypers.data_options, limit=3000 if args.data_limit > 0 else 0),
                        should_save=(hypers.global_rank == 0))
    else:
        dev_dataset, qid2sfs = None, None
    logger.info(f'loaded datasets in {time.time()-start_time} seconds')

    # if we are doing a fold, then we will take a subset of train by qid
    # get all qids from train_dataset, then separate them into train_dataset and dev_dataset
    if args.fold >= 0:
        num_folds = 5
        qids = list(set([f[0] for f in train_dataset]))
        qids.sort()
        logger.info(f'from {len(train_dataset)} instances for {len(qids)} questions')
        random.Random(789).shuffle(qids)
        apply_qids = set(qids[args.fold::num_folds])
        orig_train_dataset = train_dataset
        train_dataset = [f for f in train_dataset if f[0] not in apply_qids]
        logger.info(f'setting train data to fold {args.fold}, '
                    f'{len(train_dataset)} instances for {len(set([f[0] for f in train_dataset]))} questions')
        if args.load_model:
            # we load model in folds because we want to apply only
            validate(hypers, model, dev_dataset, qid2sfs, prediction_file=args.prediction_file)
            return
        if not args.no_validate:
            dev_dataset = [f for f in orig_train_dataset if f[0] in apply_qids]
            qid2sfs = {qid: sfs for (qid, sfs) in train_qid2sfs.items() if qid in apply_qids}
            assert qid2sfs.keys() == apply_qids
            logger.info(f'setting validation data to fold {args.fold}, '
                        f'{len(dev_dataset)} instances for {len(qid2sfs)} questions')

    # sorting needed?
    if dev_dataset is not None:
        dev_dataset.sort()
    if train_dataset is not None:
        train_dataset.sort()

    if train_dataset is None:
        validate(hypers, model, dev_dataset, qid2sfs, prediction_file=args.prediction_file)
        return

    # make optimizer
    num_train_steps = int(math.ceil(len(train_dataset) / (hypers.world_size * hypers.gradient_accumulation_steps))) * hypers.epochs
    optimizer = Optimizer(hypers, model, 0, num_train_steps)

    train_ragged = True  # whether to skip instances that don't divide evenly over the world_size
    # this many don't divide evenly by world_size
    dataset_leftovers = len(train_dataset) % hypers.world_size
    dataset_offset = dataset_leftovers if train_ragged else 0

    reporting = Reporting()
    iterations = 0
    epoch_scores = []
    for epoch in range(hypers.epochs):
        logger.info(f'On epoch {epoch}')
        # CONSIDER: sort here?
        random.Random(hypers.seed+137*epoch).shuffle(train_dataset)
        model.train()
        optimizer.reset()
        # take subset of train_dataset based on hypers.global_rank and hypers.world_size
        for batch in train_dataset[dataset_offset + hypers.global_rank::hypers.world_size]:
            loss = model(batch_to_device(batch))
            reporting.moving_averages(loss=loss.item())
            optimizer.step_loss(loss)
            iterations += 1
            if reporting.is_time():
                reporting.display()
                logger.info(f'{(hypers.world_size*iterations)} instances, '
                            f'{(hypers.world_size*iterations)/reporting.elapsed_seconds()} instances per second')

        # do dummy batches on the nodes that don't have a leftover, just make sure the gradient is zero
        if not train_ragged and dataset_leftovers > 0 and hypers.global_rank >= dataset_leftovers:
            loss = model(batch_to_device(train_dataset[hypers.global_rank]))
            loss = 0 * loss
            optimizer.step_loss(loss)

        # validate after every epoch
        if dev_dataset is not None:
            max_f1, max_em, best_thresh, scores = validate(hypers, model, dev_dataset, qid2sfs,
                                                   prediction_file=args.prediction_file)
            epoch_scores.append({'max_f1': max_f1, 'em': max_em,
                                 'best_threshold': best_thresh,
                                 })  # 'threshold_scores': scores

    # save model (just the model. no need for a multi-part checkpoint)
    if hypers.global_rank != 0:
        return
    if args.save_model:
        os.makedirs(os.path.split(args.save_model)[0], exist_ok=True)
        torch.save(model.module.state_dict(), args.save_model)
    if args.results_dir:
        hypers.write_results_file(args.results_dir,
                                  f1s=epoch_scores,
                                  instances_per_second=(hypers.world_size*iterations)/reporting.elapsed_seconds(),
                                  f1=max([es['max_f1'] for es in epoch_scores]))


if __name__ == "__main__":
    main()
