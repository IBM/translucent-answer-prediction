import logging
import numpy as np
import argparse
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from tap2.bottom_machine.hypers import DataOptions, SentMarkerStyle

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class Example:
    """
    A single training/test example for the supporting fact prediction.
    The key concepts here are 'chunks' these are chunks of retrieved passages.
    If we have short passages, we put them together to create a chunk.
    If we have long passages, we split them into multiple chunks.
    A single question may also be split into multiple examples, since our max_seq_length and max_chunks may not permit
    all passages plus the question to be in a single example together.
    """
    def __init__(self,
                 id,
                 chunk_tokens,
                 question_len,
                 sent_starts,
                 sent_ends,
                 sent_ids,
                 supporting_facts):
        self.id = id
        self.chunk_tokens = chunk_tokens  # chunks x seq_length
        self.chunk_lengths = [(len(ct)-question_len) for ct in chunk_tokens]
        self.question_len = question_len  # scalar, each chunk has a question this long prepended
        self.sent_starts = sent_starts  # num_sents
        self.sent_ends = sent_ends
        self.sent_ids = sent_ids
        self.supporting_facts = supporting_facts  # sent_ids that should be selected
        # after taking cat([ct[question_len:] for ct in chuck_tokens])
        # sent_starts and sent_ends contain the start and end token for each sentence
        self.segment_ids = [[0]*question_len + [1]*(len(ct)-question_len) for ct in chunk_tokens]
        assert len(self.chunk_tokens) == len(self.chunk_lengths)

    def display(self):
        logger.info(f'{self.id}')
        for ct, sids in zip(self.chunk_tokens, self.segment_ids):
            logger.info(f'{str(list(zip(ct, sids)))}')
        allt = [t for ct in self.chunk_tokens for t in ct[self.question_len:]]
        for si in range(len(self.sent_ids)):
            logger.info(f'{self.sent_ids[si]} = {str(allt[self.sent_starts[si]:self.sent_ends[si]+1])}')
        logger.info(f'supporting facts = {str(self.supporting_facts)}')


class ExampleBuilder:
    """
    builds Example instances from a sequence of sub-passages
    """
    def __init__(self, id, qtoks, supporting_facts, max_seq_length, max_chunks, sent_marker_style: SentMarkerStyle):
        self.id = id  # the question id
        self.qtoks = qtoks  # the question tokens
        self.supporting_facts = supporting_facts  # ids of the sentences that are supporting facts
        self.qlen = len(qtoks) + 2  # length of the question plus [CLS] and [SEP]
        self.max_seq_length = max_seq_length
        self.max_chunks = max_chunks
        self.sent_marker_style = sent_marker_style
        self.examples = []  # current set of examples constructed
        self.reset()

    def reset(self):
        self.global_sent_start_indices = []
        self.global_sent_end_indices = []
        self.sum_passage_len = 0  # total length of the chunks, excluding the qlen
        self.all_sent_ids = []
        self.chunk_tokens = []
        self.cur_chunk = None

    def add_example(self):
        if self.cur_chunk is not None:
            # add current chunk to the example if we have one
            self.cur_chunk.append('[SEP]')
            self.chunk_tokens.append(self.cur_chunk)
            self.cur_chunk = None
        assert len(self.chunk_tokens) <= self.max_chunks
        if len(self.chunk_tokens) > 0:
            ex = Example(self.id, self.chunk_tokens, self.qlen,
                    self.global_sent_start_indices, self.global_sent_end_indices, self.all_sent_ids,
                    self.supporting_facts)
            self.examples.append(ex)
            self.reset()

    def add_sub_passage(self, sp):
        passage = []
        sent_start_indices = []
        sent_end_indices = []
        sent_ids = []
        for sid, sent in sp:
            # if using sent_markers only as markers we put len(passage)+1 and len(passage)-2
            offset = 1 if self.sent_marker_style == SentMarkerStyle.mark else 0
            sent_start_indices.append(len(passage) + offset)
            passage.extend(sent)
            sent_end_indices.append(len(passage) - 1 - offset)
            sent_ids.append(sid)
        # now we've got the passage in our local vars
        if self.cur_chunk is not None and len(self.cur_chunk) + len(passage) >= self.max_seq_length:
            # finish current chunk if needed
            self.cur_chunk.append('[SEP]')
            self.sum_passage_len += 1
            self.chunk_tokens.append(self.cur_chunk)
            self.cur_chunk = None
        if len(self.chunk_tokens) >= self.max_chunks:
            # finish current example if needed
            self.add_example()
        if self.cur_chunk is None:
            # initialize current chunk if needed
            self.cur_chunk = []
            self.cur_chunk.append('[CLS]')
            self.cur_chunk.extend(self.qtoks)
            self.cur_chunk.append('[SEP]')
        if self.qlen + len(passage) > self.max_seq_length:
            raise ValueError
        # add to current example
        self.cur_chunk.extend(passage)
        self.global_sent_start_indices.extend([i + self.sum_passage_len for i in sent_start_indices])
        self.global_sent_end_indices.extend([i + self.sum_passage_len for i in sent_end_indices])
        self.all_sent_ids.extend(sent_ids)
        self.sum_passage_len += len(passage)


def to_passages(contexts, tokenizer, max_sent_len, sent_marker_style: SentMarkerStyle):
    """
    tokenize the sentences in the json example
    :param contexts: list of pairs first is passage id, second is list of sentences
    :param tokenizer:
    :param max_sent_len:
    :return:
    """
    passages = []
    if sent_marker_style == SentMarkerStyle.no:
        max_real_sent_len = max_sent_len
    else:
        max_real_sent_len = max_sent_len - 2
    for context in contexts:
        pid = context[0]
        sents = context[1]
        sidtoks = []
        for ndx, sent in enumerate(sents):
            if len(sent.strip()) == 0:  # this is quite common, lots of empty sentences
                continue
            sid = pid + ":" + str(ndx)
            stoks = tokenizer.tokenize(sent)
            if not stoks:
                logger.warning(f'No tokens for {sid}:"{sent}"')
                continue
            if len(stoks) > max_real_sent_len:
                stoks = stoks[0:max_real_sent_len]
            if sent_marker_style != SentMarkerStyle.no:
                stoks = ['[STARTSENT]'] + stoks + ['[ENDSENT]']
            assert 0 < len(stoks) <= max_sent_len
            sidtoks.append((sid, stoks))
        passages.append(sidtoks)
    return passages


def to_sub_passages(passages, qlen, max_seq_len):
    """
    split passages that are too long into multiple passages
    :param passages:
    :param qlen:
    :param max_seq_len:
    :return:
    """
    passages.sort(key=lambda p: sum([len(s[1]) for s in p]))
    sub_passages = []
    for passage in passages:
        splen = 0
        sub_passage = []
        for si in range(len(passage)):
            sent = passage[si]
            # if this sentence will make the passage too long, stop adding and make a sub-passage
            if splen + len(sent[1]) + qlen >= max_seq_len:
                sub_passages.append(sub_passage)
                # the next sub-passage will include the prev sentence
                assert si > 0  # no two sentences should make a passage too long - ensured by max_sent_len
                sub_passage = [passage[si-1]]
                splen = len(passage[si-1][1])
            sub_passage.append(sent)
            splen += len(sent[1])
        sub_passages.append(sub_passage)
    return sub_passages


def truncate_passages(passages, qlen, max_seq_len):
    passages.sort(key=lambda p: sum([len(s[1]) for s in p]))
    sub_passages = []
    for passage in passages:
        splen = 0
        sub_passage = passage
        for si in range(len(passage)):
            sent = passage[si]
            # if this sentence will make the passage too long, stop adding and make a sub-passage
            if splen + len(sent[1]) + qlen >= max_seq_len:
                sub_passage = passage[0:si]
                break
            splen += len(sent[1])
        sub_passages.append(sub_passage)
    return sub_passages


def example_to_features(jex, tokenizer, opts: DataOptions):
    """
    split the original json example into the tensor precursors for the supporting facts model
    :param jex:
    :param tokenizer:
    :param opts
    :return:
    """
    id = jex['_id']  # if '_id' in jex else 'dummyid' (a missing id is actually an error)
    question = jex['question']
    qtoks = tokenizer.tokenize(question)
    if len(qtoks) > opts.max_question_len:
        qtoks = qtoks[0:opts.max_question_len]
    supporting_facts = None
    if 'supporting_facts' in jex:
        supporting_facts = [sp[0]+':'+str(sp[1]) for sp in jex['supporting_facts']]
    contexts = jex['context']
    qlen = len(qtoks) + 2
    passages = to_passages(contexts, tokenizer, (opts.max_seq_len-1-qlen)//2, opts.sent_marker_style)
    if opts.truncate_passages:
        sub_passages = truncate_passages(passages, qlen, opts.max_seq_len)
    else:
        sub_passages = to_sub_passages(passages, qlen, opts.max_seq_len)
    example_builder = ExampleBuilder(id, qtoks, supporting_facts,
                                     opts.max_seq_len, opts.num_para_chunks, opts.sent_marker_style)
    for sp in sub_passages:
        example_builder.add_sub_passage(sp)
    example_builder.add_example()  # add final example if available
    return example_builder.examples


def get_features(filename, tokenizer: BertTokenizer, opts: DataOptions):
    with open(filename, 'r') as fp:
        jdata = json.load(fp)
    split_question_count = 0
    for jex in jdata:
        list_of_features = example_to_features(jex, tokenizer, opts)
        if len(list_of_features) > 1:
            split_question_count += 1
        if not opts.split_questions:
            list_of_features = list_of_features[0:1]
        for features in list_of_features:
            yield features
    logger.info(f'number of split questions in {filename} = {split_question_count}')


def get_data(filename, tokenizer: BertTokenizer, opts: DataOptions, limit=0):
    dataset = []
    max_chunks = 0
    max_sent_len = 0
    qid2supportingfacts = dict()
    qid2sentids = dict()
    for features in get_features(filename, tokenizer, opts):
        # convert to torch tensors
        slen = max([len(ct) for ct in features.chunk_tokens])
        chunk_token_ids = [tokenizer.convert_tokens_to_ids(ct) + [0]*(slen-len(ct)) for ct in features.chunk_tokens]
        segment_ids = [sids + [0]*(slen-len(sids)) for sids in features.segment_ids]
        if len(features.chunk_tokens) > max_chunks:
            max_chunks = len(features.chunk_tokens)
        max_sent_len = max(max_sent_len, (np.array(features.sent_ends)-np.array(features.sent_starts)).max())
        sent_targets = None
        if features.supporting_facts is not None:
            qid2supportingfacts[features.id] = features.supporting_facts
            for sid in features.sent_ids:
                qid2sentids.setdefault(features.id, set()).add(sid)
            sent_targets = torch.zeros(len(features.sent_ids), dtype=torch.float)
            for sf in features.supporting_facts:
                if sf not in features.sent_ids:
                    continue
                sent_targets[features.sent_ids.index(sf)] = 1
        assert len(features.sent_starts) == len(features.sent_ends) == len(features.sent_ids)
        assert len(chunk_token_ids) == len(features.chunk_lengths)
        dataset.append((features.id, features.sent_ids, features.question_len, features.chunk_lengths,
                        torch.tensor(chunk_token_ids, dtype=torch.long),
                        torch.tensor(segment_ids, dtype=torch.long),
                        torch.tensor(features.sent_starts, dtype=torch.long),
                        torch.tensor(features.sent_ends, dtype=torch.long),
                        sent_targets))
        if 0 < limit <= len(dataset):
            break
        if len(dataset) % 5000 == 0:
            logger.info(f'loading dataset item {len(dataset)} from {filename}')
    logger.info(f'in {filename}: max_chunks = {max_chunks}, max_sent_length = {max_sent_len}')
    out_of_recall = 0
    total_positives = 0
    for id, sps in qid2supportingfacts.items():
        total_positives += len(sps)
        sent_ids = qid2sentids.get(id)
        for sp in sps:
            if sp not in sent_ids:
                out_of_recall += 1
    if len(qid2supportingfacts) > 0:
        logger.info(f'in {filename}, due to truncations we have lost {out_of_recall} out of {total_positives} positives')
    return dataset, qid2supportingfacts


def display_batch(batch, tokenizer: BertTokenizer):
    id, sent_ids, qlen, chunk_lengths, chunk_tokens, segment_ids, sent_starts, sent_ends, sent_targets = batch
    assert chunk_tokens.shape[0] == len(chunk_lengths)
    chunk_tokens = chunk_tokens.numpy()
    segment_ids = segment_ids.numpy()
    sent_starts = sent_starts.numpy()
    sent_ends = sent_ends.numpy()
    sent_targets = sent_targets.numpy()
    logger.info(f'{id}')
    all_toks = []
    for ci in range(len(chunk_lengths)):
        clen = chunk_lengths[ci]
        chunk_toks = tokenizer.convert_ids_to_tokens(chunk_tokens[ci])
        all_toks.extend(chunk_toks[qlen:qlen+clen])
        segments = segment_ids[ci]
        logger.info(f'{str(list(zip(chunk_toks, segments)))}')
    for si in range(len(sent_ids)):
        logger.info(f'{sent_targets[si]} {sent_ids[si]} = {str(all_toks[sent_starts[si]:sent_ends[si]+1])}')


def main():
    parser = argparse.ArgumentParser()

    # Other parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="BERT model to use")
    parser.add_argument("--data", default=None, type=str, required=True,
                        help="HotpotQA json dataset")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model,
                                              do_lower_case=args.bert_model.endswith('uncased'),
                                              add_special_tokens=['[STARTSENT]', '[ENDSENT]'])

    opts = DataOptions()
    limit = 10
    for example in get_features(args.data, tokenizer,opts):
        example.display()
        limit -= 1
        if limit <= 0:
            break

    logger.info('='*80)

    for batch in get_data(args.data, tokenizer, opts, 10):
        display_batch(batch, tokenizer)


if __name__ == "__main__":
    main()
