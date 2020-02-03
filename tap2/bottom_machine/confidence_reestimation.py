import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


# rank bonuses do not appear to help at all (for macro F1. they do help for micro)
def validate_score(qid2sfs, qid2sid2score):
    # CONSIDER: do rank-aware max-F1 to get good candidate thresholds,
    # then we can do a search around the selected threshold values
    # NOTE: the rank bonuses do almost nothing
    start_time = time.time()
    logger.info(f'with rank bonus = [0.2, 0.1]')
    max_f1, max_em, best_thresh, score_list = _validate_score(qid2sfs, qid2sid2score, rank_bonuses=[0.2, 0.1])
    for rb in [[], [0.15, 0.05], [0.2, 0.1, 0.05]]:
        logger.info(f'with rank bonuses {str(rb)}')
        _validate_score(qid2sfs, qid2sid2score, rank_bonuses=rb)
    logger.info(f'scoring time = {time.time()-start_time}')
    return max_f1, max_em, best_thresh, score_list


def _validate_score(qid2sfs, qid2sid2score, rank_bonuses=None):
    # find the simple threshold that maximizes F1
    all_scores = []
    all_targets = []
    for id, sid2score in qid2sid2score.items():
        sfs = qid2sfs[id]
        # add scores for out-of-recall supporting facts
        for sf in sfs:
            if sf not in sid2score:
                sid2score[sf] = -1000
        scores = np.zeros(len(sid2score), dtype=np.float32)
        targets = np.zeros(len(sid2score), dtype=np.float32)
        ndx = 0
        for sid, score in sid2score.items():
            scores[ndx] = score
            targets[ndx] = 1 if sid in sfs else 0
            ndx += 1
        if rank_bonuses:
            sorted_ndxs = scores.argsort()[::-1]
            for r, b in enumerate(rank_bonuses):
                if r < len(sorted_ndxs):
                    scores[sorted_ndxs[r]] += b
        all_scores.append(scores)
        all_targets.append(targets)
    # the simple algorithm for Max F1
    max_f1 = 0
    max_em = 0
    best_thresh = None
    scores_per_threshold = []
    for ti in range(100):
        thresh = ti * 0.01
        sum_prec = 0
        sum_recall = 0
        sum_f1 = 0
        sum_em = 0
        for scores, targets in zip(all_scores, all_targets):
            selected = scores >= thresh
            overlap = np.count_nonzero(targets[selected])
            scount = np.count_nonzero(selected)
            p = overlap / scount if scount > 0 else 0
            r = overlap / np.count_nonzero(targets)
            sum_prec += p
            sum_recall += r
            if p > 0 and r > 0:
                sum_f1 += 2.0 / (1.0 / p + 1.0 / r)
            if p == r == 1:
                sum_em += 1

        p = sum_prec / len(all_scores)
        r = sum_recall / len(all_scores)
        em = sum_em / len(all_scores)
        f1 = sum_f1 / len(all_scores)
        scores_per_threshold.append((f1, p, r, em, thresh))
        if f1 > max_f1:
            max_f1 = f1
            max_em = em
            best_thresh = thresh
    logger.info(f'Max F1 = {max_f1}, EM = {max_em} @ {best_thresh}')
    # remove the really weak thresholds
    scores_per_threshold = [(f1, p, r, em, thresh) for (f1, p, r, em, thresh) in scores_per_threshold if f1 >= 0.8 * max_f1]
    scores_per_threshold.sort(reverse=True)

    return max_f1, max_em, best_thresh, scores_per_threshold


def fraction_full_recall(qid2sfs, qid2sid2score):
    # show what fraction of instances we have full recall on as a function of threshold
    all_scores = []
    all_targets = []
    for id, sid2score in qid2sid2score.items():
        sfs = qid2sfs[id]
        # add scores for out-of-recall supporting facts
        for sf in sfs:
            if sf not in sid2score:
                sid2score[sf] = -1000
        scores = np.zeros(len(sid2score), dtype=np.float32)
        targets = np.zeros(len(sid2score), dtype=np.float32)
        ndx = 0
        for sid, score in sid2score.items():
            scores[ndx] = score
            targets[ndx] = 1 if sid in sfs else 0
            ndx += 1
        all_scores.append(scores)
        all_targets.append(targets)
    # the simple algorithm for Max F1
    scores_per_threshold = []
    for thresh in [0.05, 0.075, 0.1, 0.2, 0.3, 0.4]:
        sum_full_recall = 0
        for scores, targets in zip(all_scores, all_targets):
            selected = scores >= thresh
            overlap = np.count_nonzero(targets[selected])
            scount = np.count_nonzero(selected)
            p = overlap / scount if scount > 0 else 0
            r = overlap / np.count_nonzero(targets)
            if r == 1:
                sum_full_recall += 1
        full_recall = sum_full_recall / len(all_scores)
        scores_per_threshold.append((full_recall, thresh))
    logger.info(f'{str(scores_per_threshold)}')


def sf_stats(qid2sid2score):
    # show what fraction of instances we have full recall on as a function of threshold
    all_scores = []
    for id, sid2score in qid2sid2score.items():
        scores = np.zeros(len(sid2score), dtype=np.float32)
        ndx = 0
        for sid, score in sid2score.items():
            scores[ndx] = score
            ndx += 1
        all_scores.append(scores)
    for thresh in [0.1, 0.41]:
        sum_sf_count = 0
        min_sf_count = 1000
        max_sf_count = 0
        for scores in all_scores:
            selected = scores >= thresh
            scount = np.count_nonzero(selected)
            sum_sf_count += scount
            min_sf_count = min(min_sf_count, scount)
            max_sf_count = max(max_sf_count, scount)
        logger.info(f'at {thresh}, supporting fact count avg = {sum_sf_count/len(all_scores)}, '
                    f'min = {min_sf_count}, max = {max_sf_count}')
